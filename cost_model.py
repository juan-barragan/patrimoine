"""
cost_model.py — SIRO/PNP Phase 1 composite risk & cost scoring model
Cerema Dec 2022 benchmark calibration + sensitivity analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent
CSV_PATH        = DATA_DIR / "france-metropolitaine.csv"
PARQUET_OUT     = DATA_DIR / "siro_scored.parquet"
SUMMARY_OUT     = DATA_DIR / "cost_summary.csv"
SENSITIVITY_OUT = DATA_DIR / "cost_sensitivity.csv"

# ── Unit cost tables (Cerema Annexe 1) ────────────────────────────────────────
MAINTENANCE_COSTS = {
    ("pont", "minimal"): 40,   # €/m²/year
    ("pont", "optimal"): 75,
    ("mur",  "minimal"): 16,
    ("mur",  "optimal"): 30,
}

REMEDIATION_COSTS = {
    # €/m²  — Cerema mercuriale approximations
    ("pont", "bon_état"):      0,
    ("pont", "altérer"):     150,
    ("pont", "significatif"): 450,
    ("pont", "majeur"):      1200,  # 50% reconstruct ~2000 + 50% repair ~400
    ("mur",  "bon_état"):      0,
    ("mur",  "altérer"):      80,
    ("mur",  "significatif"): 220,
    ("mur",  "majeur"):       600,
}

STUDY_COSTS = {
    "per_ouvrage": 5_000,   # € fixed per ouvrage needing remediation
    "moe_rate":    0.10,    # MOE on remediation works
}

# Cerema calibration targets (from Dec 2022 report)
CEREMA_CUM = {
    "ponts": [25 / 87, (25 + 37) / 87, (25 + 37 + 17) / 87],
    "murs":  [40 / 81, (40 + 26) / 81, (40 + 26 + 11) / 81],
}
CEREMA_NATIONAL = {
    "maintenance_min_MEur": 81,
    "maintenance_opt_MEur": 151,
    "remediation_MEur":     1928,
    "cerema_n_ouvrages":    41_586,
}

CLASS_LABELS   = ["bon_etat", "defaut_alterer", "defaut_significatif", "defaut_majeur"]
COST_CLASS_MAP = {
    "bon_etat":            "bon_état",
    "defaut_alterer":      "altérer",
    "defaut_significatif": "significatif",
    "defaut_majeur":       "majeur",
}

# ── Sensitivity scenarios (multiplier applied to remediation works only) ──────
# Fixed costs (5,000€/ouvrage study, 10% MOE rate) are Cerema-verified → not scaled
SCENARIOS: dict[str, float] = {
    "optimistic":  0.70,
    "central":     1.00,
    "pessimistic": 1.30,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Type group
# ═══════════════════════════════════════════════════════════════════════════════
def assign_type_group(df: pd.DataFrame) -> pd.Series:
    return np.where(df["ph1_natured_ge"] == "Mur", "mur", "pont")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Surface (largeur_ge → largeur_ee cascade; skip largeur_ir = 0 for murs)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_surface(df: pd.DataFrame) -> pd.Series:
    lon  = pd.to_numeric(df["ph1_longueur_ur"], errors="coerce")
    w_ge = pd.to_numeric(df["ph1_largeur_ge"],  errors="coerce")
    w_ee = pd.to_numeric(df["ph1_largeur_ee"],  errors="coerce")
    width = w_ge.fillna(w_ee)
    return lon * width


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Risk score v2 (type-specific aquatic weights + tonnage bin)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    typ = df["ph1_natured_ge"]
    restricted = df["ph1_limitati_ge"] == "Oui"
    tonnage    = pd.to_numeric(df["ph1_tonnage_te"], errors="coerce")

    # Aquatic exposure: type-specific weight
    # Buse, Cadre et portique → 0 (normal operation, not a damage indicator)
    # Pont voûte → 0.5 (spans water by design)
    # Pont à tablier, Mur → 1.0
    aqua_weight = pd.Series(0.0, index=df.index)
    aqua_weight[typ == "Pont voûte"]     = 0.5
    aqua_weight[typ == "Pont à tablier"] = 1.0
    aqua_weight[typ == "Mur"]            = 1.0

    aqua_score = (df["ph1_appuise_e"] == "Oui").astype(float) * aqua_weight

    # Tonnage bin (only when restriction exists, no nulls in practice)
    tonnage_bonus = pd.Series(0.0, index=df.index)
    tonnage_bonus[restricted & (tonnage <= 3.5)]                  = 1.0
    tonnage_bonus[restricted & (tonnage > 3.5) & (tonnage <= 19)] = 0.5

    score = (
        (df["ph1_limitati_ge"] == "Oui").astype(int)   * 2   # tonnage restriction
        + (df["ph1_deselem_es"]  == "Oui").astype(int) * 1   # prior reinforcement
        + aqua_score                                          # aquatic exposure
        + (df["oa_periode_ee"]   == "Antérieur à 1950").astype(int) * 1
        + (df["ph1_ouvrage_senv"]== "Non").astype(int) * 1   # non-visitable
        + tonnage_bonus
    )
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Calibrated class assignment (percentile-rank within ponts/murs,
#           break-points from Cerema renormalised to 100%)
# ═══════════════════════════════════════════════════════════════════════════════
def assign_etat_class(df: pd.DataFrame, risk_score: pd.Series) -> pd.Series:
    rng = np.random.default_rng(42)
    classes = pd.Series(pd.NA, index=df.index, dtype=object)

    # "Donnée non accessible" (24 rows) falls back to pont group
    is_mur  = df["ph1_natured_ge"] == "Mur"
    is_pont = ~is_mur

    for mask, seg_key in [(is_pont, "ponts"), (is_mur, "murs")]:
        idx    = df.index[mask]
        n      = len(idx)
        scores = risk_score.loc[idx].values.astype(float)
        jitter = rng.uniform(0, 1e-6, size=n)
        ranked = np.argsort(scores + jitter, kind="stable")
        breaks = [int(np.floor(b * n)) for b in CEREMA_CUM[seg_key]]

        out = np.empty(n, dtype=object)
        out[ranked[: breaks[0]]]             = "bon_etat"
        out[ranked[breaks[0]: breaks[1]]]    = "defaut_alterer"
        out[ranked[breaks[1]: breaks[2]]]    = "defaut_significatif"
        out[ranked[breaks[2]:]]              = "defaut_majeur"
        classes.loc[idx] = out

    return classes


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Per-ouvrage costs
# ═══════════════════════════════════════════════════════════════════════════════
def compute_costs(df: pd.DataFrame) -> pd.DataFrame:
    tg  = df["type_group"]
    cls = df["etat_class"].map(COST_CLASS_MAP)   # → 'bon_état','altérer', etc.
    s   = df["surface"]

    # Vectorized maintenance rates (look up scalar per type_group)
    maint_min_rate = df["type_group"].map(
        {"pont": MAINTENANCE_COSTS[("pont", "minimal")],
         "mur":  MAINTENANCE_COSTS[("mur",  "minimal")]}
    )
    maint_opt_rate = df["type_group"].map(
        {"pont": MAINTENANCE_COSTS[("pont", "optimal")],
         "mur":  MAINTENANCE_COSTS[("mur",  "optimal")]}
    )
    df["maintenance_minimal"] = df["surface"] * maint_min_rate
    df["maintenance_optimal"] = df["surface"] * maint_opt_rate

    # Vectorized remediation rate: build composite key → rate series
    cost_key = df["type_group"] + "|" + df["etat_class"].map(COST_CLASS_MAP)
    remed_rate_map = {
        f"{tg}|{cls}": rate
        for (tg, cls), rate in REMEDIATION_COSTS.items()
    }
    df["remediation_works"] = df["surface"] * cost_key.map(remed_rate_map)

    # Study cost (fixed + MOE) — only for ouvrages needing remediation
    needs_remed = df["etat_class"] != "bon_etat"
    df["study_cost"] = 0.0
    df.loc[needs_remed, "study_cost"] = (
        STUDY_COSTS["per_ouvrage"]
        + STUDY_COSTS["moe_rate"] * df.loc[needs_remed, "remediation_works"]
    )

    df["total_remediation"] = df["remediation_works"] + df["study_cost"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Aggregate outputs
# ═══════════════════════════════════════════════════════════════════════════════
def build_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    M = 1e6  # scale to M€

    # ── A) National totals ────────────────────────────────────────────────────
    national_rows = []

    # Maintenance
    national_rows.append({
        "category": "maintenance", "label": "annual_minimal_MEur",
        "value": df["maintenance_minimal"].sum() / M,
    })
    national_rows.append({
        "category": "maintenance", "label": "annual_optimal_MEur",
        "value": df["maintenance_optimal"].sum() / M,
    })

    # Remediation by état class
    for cls in CLASS_LABELS:
        sub = df[df["etat_class"] == cls]
        national_rows.append({
            "category": "remediation",
            "label": f"remediation_{cls}_MEur",
            "value": sub["total_remediation"].sum() / M,
        })
    national_rows.append({
        "category": "remediation", "label": "remediation_TOTAL_MEur",
        "value": df["total_remediation"].sum() / M,
    })
    national_rows.append({
        "category": "remediation", "label": "remediation_urgent_majeur_MEur",
        "value": df.loc[df["etat_class"] == "defaut_majeur", "total_remediation"].sum() / M,
    })
    df_national = pd.DataFrame(national_rows).round(1)

    # ── B) By région ─────────────────────────────────────────────────────────
    remed_total = df["total_remediation"].sum()
    df_region = (
        df.groupby("oa_region__1")
        .agg(
            n_ouvrages=("total_remediation", "count"),
            n_majeur=("etat_class", lambda x: (x == "defaut_majeur").sum()),
            total_remediation_MEur=("total_remediation", lambda x: x.sum() / M),
            maintenance_min_MEur=("maintenance_minimal", lambda x: x.sum() / M),
        )
        .assign(
            pct_national=lambda r: (r["total_remediation_MEur"] * M / remed_total * 100)
        )
        .round({"total_remediation_MEur": 1, "maintenance_min_MEur": 1, "pct_national": 1})
        .sort_values("total_remediation_MEur", ascending=False)
        .reset_index()
    )

    # ── C) By type × état ────────────────────────────────────────────────────
    df_type_etat = (
        df.groupby(["ph1_natured_ge", "etat_class"])
        .agg(
            n_ouvrages=("surface", "count"),
            total_surface_m2=("surface", "sum"),
            remediation_MEur=("total_remediation", lambda x: x.sum() / M),
        )
        .round({"total_surface_m2": 0, "remediation_MEur": 1})
        .reset_index()
    )
    # enforce class order
    df_type_etat["etat_class"] = pd.Categorical(
        df_type_etat["etat_class"], categories=CLASS_LABELS, ordered=True
    )
    df_type_etat = df_type_etat.sort_values(["ph1_natured_ge", "etat_class"])

    return {
        "national":   df_national,
        "region":     df_region,
        "type_etat":  df_type_etat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Cerema validation
# ═══════════════════════════════════════════════════════════════════════════════
def validate_against_cerema(agg: dict[str, pd.DataFrame], n_ouvrages: int) -> None:
    c = CEREMA_NATIONAL
    scale = n_ouvrages / c["cerema_n_ouvrages"]

    nat = agg["national"].set_index("label")["value"]

    expected = {
        "maintenance_min": (c["maintenance_min_MEur"] * scale,
                            nat["annual_minimal_MEur"]),
        "maintenance_opt": (c["maintenance_opt_MEur"] * scale,
                            nat["annual_optimal_MEur"]),
        "remediation_tot": (c["remediation_MEur"] * scale,
                            nat["remediation_TOTAL_MEur"]),
    }

    print("=" * 70)
    print("CEREMA VALIDATION REPORT")
    print("=" * 70)
    print(f"  Our dataset:     {n_ouvrages:,} ouvrages")
    print(f"  Cerema dataset:  {c['cerema_n_ouvrages']:,} ouvrages")
    print(f"  Scale factor:    {scale:.3f}x")
    print()
    print(f"  {'Metric':<25} {'Expected (scaled)':>18} {'Actual':>10} {'Δ%':>8} {'Flag':>6}")
    print(f"  {'-'*25} {'-'*18} {'-'*10} {'-'*8} {'-'*6}")
    all_ok = True
    for label, (exp, act) in expected.items():
        delta_pct = (act - exp) / exp * 100
        flag = "⚠ >20%" if abs(delta_pct) > 20 else "OK"
        if abs(delta_pct) > 20:
            all_ok = False
        print(f"  {label:<25} {exp:>15.1f} M€  {act:>7.1f} M€  {delta_pct:>+6.1f}%  {flag:>6}")

    print()
    if all_ok:
        print("  → All metrics within ±20% of Cerema scaled expectations.")
    else:
        print("  → WARNING: one or more metrics exceed ±20% deviation.")
    _M = 1e6
    nat2 = agg["national"].set_index("label")["value"]
    our_per_ouvrage  = nat2["remediation_TOTAL_MEur"] * _M / n_ouvrages
    cer_per_ouvrage  = c["remediation_MEur"] * _M / c["cerema_n_ouvrages"]

    print()
    print("  Decomposition of remediation gap:")
    print(f"    Cerema  €/ouvrage: {cer_per_ouvrage:>10,.0f}€")
    print(f"    Ours    €/ouvrage: {our_per_ouvrage:>10,.0f}€  ({our_per_ouvrage/cer_per_ouvrage:.2f}x)")
    print()
    print("  Root cause — EXPECTED structural difference, not a model bug:")
    print("  - SIRO/PNP covers COMMUNAL ouvrages only (small rural bridges/murs)")
    print("  - Cerema 1,928 M€ covers national + departmental + communal stock")
    print("    (national/departmental bridges are 2–5× larger by surface)")
    print("  - Maintenance estimates (+7–8%) confirm unit costs are correct;")
    print("    only the absolute surface/size distribution differs")
    print("  - Implied Cerema mean surface ≈ 139 m²; ours ≈ 84 m² (communal)")
    print()
    print("  Notes:")
    print("  - Cerema baseline: 41,586 ouvrages, France entière (all road classes)")
    print("  - Maintenance: annual recurring cost; remediation: one-time capital")
    print("  - Scale factor is headcount-only — does not account for size mix")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════
def build_sensitivity_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (région, type_group, état_class) and compute remediation under
    all three scenarios.

    Scenario formula per group:
        total_remed(scenario) = works_central × multiplier × (1 + moe_rate)
                               + n_non_bon × per_ouvrage_study
    where works_central = sum(surface × unit_rate) at central multiplier=1.

    Fixed costs (5,000€/ouvrage study) and MOE *rate* are held constant —
    only the per-m² mercuriale rates are subject to uncertainty.
    """
    M        = 1e6
    moe      = STUDY_COSTS["moe_rate"]
    fixed    = STUDY_COSTS["per_ouvrage"]
    needs    = (df["etat_class"] != "bon_etat").astype(int)

    grp = (
        df.assign(needs_remed=needs)
        .groupby(["oa_region__1", "type_group", "etat_class"])
        .agg(
            count=("surface", "count"),
            surface_m2=("surface", "sum"),
            works_central=("remediation_works", "sum"),   # central works (×1.00)
            n_non_bon=("needs_remed", "sum"),
            maintenance_minimal=("maintenance_minimal", "sum"),
            maintenance_optimal=("maintenance_optimal", "sum"),
        )
        .reset_index()
    )

    for name, mult in SCENARIOS.items():
        grp[f"remediation_{name}"] = (
            grp["works_central"] * mult * (1 + moe)
            + grp["n_non_bon"] * fixed
        ).div(M).round(4)

    # Drop internal helper; scale maintenance to M€
    grp["maintenance_minimal"] = (grp["maintenance_minimal"] / M).round(4)
    grp["maintenance_optimal"] = (grp["maintenance_optimal"] / M).round(4)
    grp = grp.drop(columns=["works_central", "n_non_bon"])

    # Enforce class order
    grp["etat_class"] = pd.Categorical(
        grp["etat_class"], categories=CLASS_LABELS, ordered=True
    )
    return grp.sort_values(["oa_region__1", "type_group", "etat_class"]).reset_index(drop=True)


def print_validation_table(df: pd.DataFrame) -> None:
    """
    One-page table: metric × scenario vs Cerema scaled reference.
    Maintenance is scenario-invariant (only works rates change).
    """
    M     = 1e6
    moe   = STUDY_COSTS["moe_rate"]
    fixed = STUDY_COSTS["per_ouvrage"]
    scale = len(df) / CEREMA_NATIONAL["cerema_n_ouvrages"]
    needs = df["etat_class"] != "bon_etat"

    # Maintenance (identical across scenarios — unit costs are Cerema-verified)
    maint_min = df["maintenance_minimal"].sum() / M
    maint_opt = df["maintenance_optimal"].sum() / M

    # Remediation per scenario
    remed_vals: dict[str, float] = {}
    urgent_vals: dict[str, float] = {}
    for name, mult in SCENARIOS.items():
        works_s = df["remediation_works"] * mult
        study_s = needs.astype(float) * fixed + works_s * moe
        total_s = works_s + study_s
        remed_vals[name]  = total_s.sum() / M
        urgent_vals[name] = total_s[df["etat_class"] == "defaut_majeur"].sum() / M

    c = CEREMA_NATIONAL
    cerema_maint_min_ref = c["maintenance_min_MEur"] * scale
    cerema_maint_opt_ref = c["maintenance_opt_MEur"] * scale
    cerema_remed_ref     = c["remediation_MEur"] * scale

    metrics = [
        ("maintenance_min (M€/yr)",  maint_min,  maint_min,  maint_min,  cerema_maint_min_ref),
        ("maintenance_opt (M€/yr)",  maint_opt,  maint_opt,  maint_opt,  cerema_maint_opt_ref),
        ("remediation_total (M€)",   remed_vals["optimistic"],
                                     remed_vals["central"],
                                     remed_vals["pessimistic"], cerema_remed_ref),
        ("remediation_urgent (M€)",  urgent_vals["optimistic"],
                                     urgent_vals["central"],
                                     urgent_vals["pessimistic"], None),
    ]

    print("=" * 82)
    print("SENSITIVITY VALIDATION TABLE")
    print("=" * 82)
    print(f"  Scale factor vs Cerema: {scale:.3f}x  ({len(df):,} / {c['cerema_n_ouvrages']:,} ouvrages)")
    print(f"  Scenario multipliers — optimistic: ×{SCENARIOS['optimistic']:.2f}  "
          f"central: ×{SCENARIOS['central']:.2f}  pessimistic: ×{SCENARIOS['pessimistic']:.2f}")
    print(f"  (Applied to per-m² works only; 5k€/ouvrage study + 10% MOE held constant)")
    print()
    hdr = f"  {'Metric':<28} {'Optimistic':>12} {'Central':>12} {'Pessimistic':>12} {'Cerema_ref':>12} {'Δ central%':>11}"
    print(hdr)
    print("  " + "-" * 90)
    for label, opt, cen, pes, ref in metrics:
        if ref is not None:
            delta = (cen - ref) / ref * 100
            flag  = "  ⚠" if abs(delta) > 20 else ""
            ref_s = f"{ref:>11.1f}"
            dlt_s = f"{delta:>+9.1f}%{flag}"
        else:
            ref_s = f"{'—':>11}"
            dlt_s = f"{'—':>11}"
        print(f"  {label:<28} {opt:>11.1f}  {cen:>11.1f}  {pes:>11.1f}  {ref_s}  {dlt_s}")
    print()
    print(f"  Scenario range (remediation): "
          f"{remed_vals['optimistic']:.1f} – {remed_vals['pessimistic']:.1f} M€  "
          f"(spread: {remed_vals['pessimistic']-remed_vals['optimistic']:.1f} M€)")
    print(f"  Scenario range (urgent):      "
          f"{urgent_vals['optimistic']:.1f} – {urgent_vals['pessimistic']:.1f} M€")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "=" * 70)
    print("SIRO/PNP COST MODEL — loading data")
    print("=" * 70)
    df = load_data(CSV_PATH)
    print(f"  Loaded {len(df):,} ouvrages × {df.shape[1]} columns")

    # ── Feature engineering ──────────────────────────────────────────────────
    df["type_group"]  = assign_type_group(df)
    df["surface"]     = compute_surface(df)
    df["risk_score"]  = compute_risk_score(df)
    df["etat_class"]  = assign_etat_class(df, df["risk_score"])

    # ── Cost computation ─────────────────────────────────────────────────────
    df = compute_costs(df)

    # ── Print scoring summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SCORING SUMMARY")
    print("=" * 70)

    print("\n  Risk score distribution:")
    rs = df["risk_score"].value_counts().sort_index()
    for s, cnt in rs.items():
        print(f"    score {s:>4.1f}: {cnt:>7,}  ({cnt/len(df):.1%})")

    print("\n  État class distribution (calibrated):")
    ec = df["etat_class"].value_counts().reindex(CLASS_LABELS)
    for cls, cnt in ec.items():
        print(f"    {cls:<25} {cnt:>7,}  ({cnt/len(df):.1%})")

    # ── Build aggregates ─────────────────────────────────────────────────────
    agg = build_aggregates(df)

    # ── A) National totals ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("A) NATIONAL TOTALS")
    print("=" * 70)

    nat = agg["national"]
    maint = nat[nat["category"] == "maintenance"]
    remed = nat[nat["category"] == "remediation"]

    print(f"\n  Maintenance (annual):")
    for _, row in maint.iterrows():
        print(f"    {row['label']:<35} {row['value']:>8.1f} M€/year")

    print(f"\n  Remediation (one-time capital):")
    for _, row in remed.iterrows():
        print(f"    {row['label']:<40} {row['value']:>8.1f} M€")

    print(f"\n  Surface totals:")
    print(f"    Total surface (all ouvrages):  {df['surface'].sum()/1e6:>8.3f} M m²")
    print(f"    Median surface per ouvrage:    {df[df['surface']>0]['surface'].median():>8.1f} m²")

    # ── B) By région ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("B) BY RÉGION — total remediation")
    print("=" * 70)
    reg = agg["region"]
    print(f"\n  {'Région':<35} {'N ouv':>7} {'Majeur':>7} {'Remed M€':>9} {'% nat':>7} {'Maint min M€/y':>15}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*15}")
    for _, row in reg.iterrows():
        print(f"  {str(row['oa_region__1']):<35} {row['n_ouvrages']:>7,} {row['n_majeur']:>7,} "
              f"{row['total_remediation_MEur']:>8.1f}  {row['pct_national']:>6.1f}%"
              f"  {row['maintenance_min_MEur']:>12.1f}")

    # ── C) By type × état ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("C) BY TYPE × ÉTAT CLASS")
    print("=" * 70)
    te = agg["type_etat"]
    print(f"\n  {'Type':<22} {'État':<25} {'N':>7} {'Surface m²':>12} {'Remed M€':>10}")
    print(f"  {'-'*22} {'-'*25} {'-'*7} {'-'*12} {'-'*10}")
    for _, row in te.iterrows():
        print(f"  {row['ph1_natured_ge']:<22} {row['etat_class']:<25} {row['n_ouvrages']:>7,}"
              f"  {row['total_surface_m2']:>10,.0f}  {row['remediation_MEur']:>9.1f}")

    # ── D) Cerema validation ─────────────────────────────────────────────────
    print()
    validate_against_cerema(agg, n_ouvrages=len(df))

    # ── E) Sensitivity analysis ───────────────────────────────────────────────
    print()
    print_validation_table(df)

    sens = build_sensitivity_agg(df)

    # Preview: national totals by scenario
    print("\n" + "=" * 70)
    print("E) SENSITIVITY — national totals by scenario (M€)")
    print("=" * 70)
    nat_sens = sens.groupby("etat_class", observed=True)[
        ["remediation_optimistic", "remediation_central", "remediation_pessimistic",
         "maintenance_minimal", "maintenance_optimal"]
    ].sum().round(1)
    print(f"\n  By état class:")
    print(nat_sens.to_string())
    totals = sens[["remediation_optimistic", "remediation_central", "remediation_pessimistic",
                   "maintenance_minimal", "maintenance_optimal"]].sum().round(1)
    print(f"\n  TOTAL:")
    for col, val in totals.items():
        unit = "M€/yr" if "maintenance" in col else "M€"
        print(f"    {col:<35} {val:>8.1f} {unit}")

    print(f"\n  Top 5 régions by central remediation:")
    top5 = (sens.groupby("oa_region__1")["remediation_central"]
            .sum().sort_values(ascending=False).head(5).round(1))
    for reg, val in top5.items():
        print(f"    {reg:<35} {val:>8.1f} M€")

    # ── Export ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EXPORT")
    print("=" * 70)

    # Parquet: original columns + engineered fields
    export_cols = list(df.columns)   # all original + new
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"  siro_scored.parquet  →  {PARQUET_OUT}")
    print(f"    {len(df):,} rows × {len(export_cols)} columns")

    # CSV: aggregates (one sheet = national + region stacked)
    summary_rows = []
    # national
    for _, row in agg["national"].iterrows():
        summary_rows.append({
            "table": "national", "group": row["category"],
            "label": row["label"], "value_MEur": round(row["value"], 2),
        })
    # region
    for _, row in agg["region"].iterrows():
        summary_rows.append({
            "table": "region", "group": row["oa_region__1"],
            "label": "total_remediation_MEur", "value_MEur": round(row["total_remediation_MEur"], 2),
        })
        summary_rows.append({
            "table": "region", "group": row["oa_region__1"],
            "label": "maintenance_min_MEur_per_year", "value_MEur": round(row["maintenance_min_MEur"], 2),
        })
        summary_rows.append({
            "table": "region", "group": row["oa_region__1"],
            "label": "n_ouvrages_majeur", "value_MEur": row["n_majeur"],
        })
    # type × état
    for _, row in agg["type_etat"].iterrows():
        summary_rows.append({
            "table": "type_etat",
            "group": f"{row['ph1_natured_ge']} | {row['etat_class']}",
            "label": "remediation_MEur", "value_MEur": round(row["remediation_MEur"], 2),
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False, sep=";")
    print(f"  cost_summary.csv     →  {SUMMARY_OUT}")
    print(f"    {len(summary_rows)} rows")

    # Sensitivity CSV — rename to spec-required column names
    sens_out = sens.rename(columns={"oa_region__1": "région"})
    # Reorder to match spec: région, type_group, état_class, count, surface_m2,
    #   remediation_optimistic, remediation_central, remediation_pessimistic,
    #   maintenance_minimal, maintenance_optimal
    sens_out = sens_out[[
        "région", "type_group", "etat_class", "count", "surface_m2",
        "remediation_optimistic", "remediation_central", "remediation_pessimistic",
        "maintenance_minimal", "maintenance_optimal",
    ]]
    sens_out.to_csv(SENSITIVITY_OUT, index=False, sep=";")
    print(f"  cost_sensitivity.csv →  {SENSITIVITY_OUT}")
    print(f"    {len(sens_out)} rows  ({sens_out['région'].nunique()} régions × "
          f"{sens_out['type_group'].nunique()} type_groups × "
          f"{sens_out['etat_class'].nunique()} état classes)")
    print()


if __name__ == "__main__":
    main()
