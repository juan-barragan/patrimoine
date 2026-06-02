"""
csv_export.py — Export SIRO scored data to a flat CSV
Coordinate conversion: Lambert 93 (EPSG:2154) → WGS84 (EPSG:4326)
"""

from pathlib import Path

import pandas as pd
from pyproj import Transformer

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
PARQUET_IN = DATA_DIR / "siro_scored.parquet"
CSV_OUT    = DATA_DIR / "ouvrages_export.csv"

# ── Scenario multipliers (mirror cost_model.py) ────────────────────────────────
MOE_RATE = 0.10
SCENARIOS = {"optimistic": 0.70, "central": 1.00, "pessimistic": 1.30}

# ── Human-readable labels ──────────────────────────────────────────────────────
ETAT_LABEL = {
    "bon_etat":            "Bon état",
    "defaut_alterer":      "Défaut pouvant altérer",
    "defaut_significatif": "Défaut significatif",
    "defaut_majeur":       "Défaut majeur",
}


def main() -> None:
    print("=" * 60)
    print("CSV EXPORT — SIRO/PNP ouvrage scoring")
    print("=" * 60)

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"\n  Loading {PARQUET_IN.name} …")
    df = pd.read_parquet(PARQUET_IN)
    print(f"  {len(df):,} rows × {df.shape[1]} columns")

    # ── Coordinate conversion ──────────────────────────────────────────────────
    t = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    df["longitude"], df["latitude"] = t.transform(df["x"].values, df["y"].values)
    print(f"  Lambert93 → WGS84  "
          f"(lon {df['longitude'].min():.3f}–{df['longitude'].max():.3f}, "
          f"lat {df['latitude'].min():.3f}–{df['latitude'].max():.3f})")

    # ── Per-scenario remediation costs ─────────────────────────────────────────
    study_fixed = df["study_cost"] - MOE_RATE * df["remediation_works"]
    for name, mult in SCENARIOS.items():
        df[f"remediation_{name}_eur"] = (
            df["remediation_works"] * mult * (1 + MOE_RATE) + study_fixed
        ).round(2)

    # ── Build output dataframe ─────────────────────────────────────────────────
    out = pd.DataFrame({
        # Identity
        "id":                      df["id"],
        "nom_usuel":               df["oa_nomusue_el"],
        "obstacle_franchi":        df["oa_nomdel_al"],
        "commune":                 df["oa_nomcom"],
        "departement":             df["oa_departem__1"],
        "region":                  df["oa_region__1"],
        "gestionnaire":            df["oa_gestionn_re"],
        # Geometry
        "latitude":                df["latitude"].round(6),
        "longitude":               df["longitude"].round(6),
        "surface_m2":              df["surface"].round(1),
        # Type & classification
        "type_ouvrage":            df["ph1_natured_ge"],
        "periode_construction":    df["oa_periode_ee"],
        "etat_class":              df["etat_class"].map(ETAT_LABEL).fillna(df["etat_class"]),
        "risk_score":              df["risk_score"].round(2),
        # Risk flags
        "limitation_tonnage":      df["ph1_limitati_ge"],
        "tonnage_limite_t":        df["ph1_tonnage_te"],
        "appuis_aquatique":        df["ph1_appuise_e"],
        "elements_defaillants":    df["ph1_deselem_es"],
        # Costs (€)
        "maintenance_minimale_eur_an":  df["maintenance_minimal"].round(2),
        "maintenance_optimale_eur_an":  df["maintenance_optimal"].round(2),
        "remediation_optimiste_eur":    df["remediation_optimistic_eur"].round(2),
        "remediation_centrale_eur":     df["remediation_central_eur"].round(2),
        "remediation_pessimiste_eur":   df["remediation_pessimistic_eur"].round(2),
    })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    print(f"\n  Writing {CSV_OUT.name} …")
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig", sep=";")

    size_mb = CSV_OUT.stat().st_size / 1e6
    print(f"  Done → {size_mb:.1f} MB  ({len(out):,} rows × {len(out.columns)} columns)")

    # ── Quick summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n  Distribution par état :")
    for label, n in out["etat_class"].value_counts().items():
        print(f"    {label:<30} {n:>7,}  ({100*n/len(out):.1f}%)")

    print("\n  Coûts agrégés (scénario central) :")
    total_maint = out["maintenance_minimale_eur_an"].sum()
    total_remed = out["remediation_centrale_eur"].sum()
    print(f"    Maintenance minimale annuelle : {total_maint/1e6:>8.1f} M€/an")
    print(f"    Remédiation totale (central)  : {total_remed/1e6:>8.1f} M€")

    print("\n  Top 5 régions par coût de remédiation (M€, central) :")
    reg = (out.groupby("region")["remediation_centrale_eur"]
              .sum()
              .sort_values(ascending=False)
              .head(5))
    for region, val in reg.items():
        print(f"    {region:<35} {val/1e6:>8.1f} M€")

    print(f"\n  Colonnes exportées : {list(out.columns)}")
    print(f"\n  Encodage : UTF-8 BOM (lisible directement dans Excel)")
    print(f"  Séparateur : point-virgule")


if __name__ == "__main__":
    main()
