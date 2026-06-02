"""
map_export.py — Export SIRO scored data to GeoJSON + lightweight mini-JSON
Coordinate conversion: Lambert 93 (EPSG:2154) → WGS84 (EPSG:4326)
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent
PARQUET_IN     = DATA_DIR / "siro_scored.parquet"
GEOJSON_OUT    = DATA_DIR / "map_data.geojson"
MINI_OUT       = DATA_DIR / "map_data_mini.json"

# ── Scenario multipliers (mirror cost_model.py) ───────────────────────────────
MOE_RATE       = 0.10
SCENARIOS      = {"optimistic": 0.70, "central": 1.00, "pessimistic": 1.30}

# ── Encoding maps for mini format ─────────────────────────────────────────────
ETAT_INT = {
    "bon_etat":            0,
    "defaut_alterer":      1,
    "defaut_significatif": 2,
    "defaut_majeur":       3,
}
TYPE_INT = {
    "Pont à tablier":      0,
    "Pont voûte":          1,
    "Mur":                 2,
    "Buse":                3,
    "Cadre et portique":   4,
}


# ═══════════════════════════════════════════════════════════════════════════════
def load_and_enrich() -> pd.DataFrame:
    print(f"  Loading {PARQUET_IN.name} …")
    df = pd.read_parquet(PARQUET_IN)
    print(f"  {len(df):,} rows × {df.shape[1]} columns")

    # ── Coordinate conversion ─────────────────────────────────────────────────
    t = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    df["lon"], df["lat"] = t.transform(df["x"].values, df["y"].values)
    print(f"  Converted Lambert93 → WGS84  "
          f"(lon {df['lon'].min():.3f}–{df['lon'].max():.3f}, "
          f"lat {df['lat'].min():.3f}–{df['lat'].max():.3f})")

    # ── Per-scenario remediation (derived from remediation_works + study_cost) ─
    # study_cost = study_fixed (5000 if non-bon, else 0) + MOE (10% of works)
    # → study_fixed = study_cost − 0.10 × remediation_works
    study_fixed = df["study_cost"] - MOE_RATE * df["remediation_works"]

    for name, mult in SCENARIOS.items():
        df[f"remediation_{name}"] = (
            df["remediation_works"] * mult * (1 + MOE_RATE) + study_fixed
        ).round(2)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
def _safe(val):
    """Convert numpy scalars / NaN to plain Python for JSON serialisation."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def export_geojson(df: pd.DataFrame) -> None:
    print(f"\n  Writing {GEOJSON_OUT.name} …")

    # Stream-write to avoid building the full dict in memory
    with open(GEOJSON_OUT, "w", encoding="utf-8") as fh:
        fh.write('{"type":"FeatureCollection","features":[\n')
        n = len(df)
        for pos, (_, row) in enumerate(df.iterrows()):
            props = {
                "id":                    f"{_safe(row.get('id'))}_{pos}",
                "source_id":             _safe(row.get("id")),
                "commune":               _safe(row.get("oa_nomcom")),
                "departement":           _safe(row.get("oa_departem__1")),
                "region":                _safe(row.get("oa_region__1")),
                "type":                  _safe(row.get("ph1_natured_ge")),
                "periode":               _safe(row.get("oa_periode_ee")),
                "etat_class":            _safe(row.get("etat_class")),
                "risk_score":            _safe(row.get("risk_score")),
                "surface_m2":            _safe(round(float(row.get("surface", 0) or 0), 1)),
                "maintenance_min":       _safe(round(float(row.get("maintenance_minimal", 0) or 0), 2)),
                "maintenance_opt":       _safe(round(float(row.get("maintenance_optimal", 0) or 0), 2)),
                "total_remediation":     _safe(round(float(row.get("total_remediation", 0) or 0), 2)),
                "remediation_central":   _safe(round(float(row.get("remediation_central", 0) or 0), 2)),
                "remediation_pessimistic": _safe(round(float(row.get("remediation_pessimistic", 0) or 0), 2)),
                "remediation_optimistic":  _safe(round(float(row.get("remediation_optimistic", 0) or 0), 2)),
                "tonnage_restriction":   _safe(row.get("ph1_limitati_ge")),
                "tonnage_limit":         _safe(row.get("ph1_tonnage_te")),
                "aquatic":               _safe(row.get("ph1_appuise_e")),
                "prior_repairs":         _safe(row.get("ph1_deselem_es")),
                "gestionnaire":          _safe(row.get("oa_gestionn_re")),
                "nom_usuel":             _safe(row.get("oa_nomusue_el")),
                "obstacle":              _safe(row.get("oa_nomdel_al")),
            }
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        round(float(row["lon"]), 6),
                        round(float(row["lat"]), 6),
                    ],
                },
                "properties": props,
            }
            sep = ",\n" if pos < n - 1 else "\n"
            fh.write(json.dumps(feature, ensure_ascii=False) + sep)

            if (pos + 1) % 10_000 == 0:
                print(f"    {pos+1:>6,} / {n:,} features written …")

        fh.write("]}\n")

    size_mb = os.path.getsize(GEOJSON_OUT) / 1e6
    print(f"  Done → {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
def export_mini(df: pd.DataFrame) -> None:
    """
    Lightweight flat array format:
    [id, lat, lon, etat_int, type_int, surface, maint_min,
     remediation_central, risk_score]
    Coordinates rounded to 5 dp (~1m precision), costs rounded to nearest €.
    """
    print(f"\n  Writing {MINI_OUT.name} …")

    etat_col   = df["etat_class"].map(ETAT_INT).fillna(-1).astype(int)
    type_col   = df["ph1_natured_ge"].map(TYPE_INT).fillna(-1).astype(int)
    surface    = df["surface"].fillna(0).round(1)
    maint_min  = df["maintenance_minimal"].fillna(0).round(0).astype(int)
    remed_cen  = df["remediation_central"].fillna(0).round(0).astype(int)
    risk       = df["risk_score"].fillna(0).round(2)
    lat_r      = df["lat"].round(5)
    lon_r      = df["lon"].round(5)
    ids        = df["id"].fillna("").astype(str)
    communes   = df["oa_nomcom"].fillna("").astype(str)
    depts      = df["oa_departem__1"].fillna("").astype(str)
    periodes   = df["oa_periode_ee"].fillna("").astype(str)
    nom_usuel  = df["oa_nomusue_el"].fillna("").astype(str)
    remed_opt  = df["remediation_optimistic"].fillna(0).round(0).astype(int)
    remed_pes  = df["remediation_pessimistic"].fillna(0).round(0).astype(int)
    tonnage_r  = df["ph1_limitati_ge"].fillna("").astype(str)
    tonnage_l  = df["ph1_tonnage_te"].fillna(0)
    aquatic    = df["ph1_appuise_e"].fillna("").astype(str)
    repairs    = df["ph1_deselem_es"].fillna("").astype(str)

    rows = [
        [
            f"{ids.iat[i]}_{i}",
            float(lat_r.iat[i]),
            float(lon_r.iat[i]),
            int(etat_col.iat[i]),
            int(type_col.iat[i]),
            float(surface.iat[i]),
            int(maint_min.iat[i]),
            int(remed_cen.iat[i]),
            float(risk.iat[i]),
            communes.iat[i],
            depts.iat[i],
            periodes.iat[i],
            nom_usuel.iat[i],
            int(remed_opt.iat[i]),
            int(remed_pes.iat[i]),
            tonnage_r.iat[i],
            float(tonnage_l.iat[i]) if tonnage_l.iat[i] else 0,
            aquatic.iat[i],
            repairs.iat[i],
        ]
        for i in range(len(df))
    ]

    meta = {
        "description": "SIRO/PNP ouvrage scoring — lightweight map data",
        "columns":     ["id", "lat", "lon", "etat_int", "type_int",
                        "surface_m2", "maint_min_eur_yr", "remediation_central_eur",
                        "risk_score", "commune", "departement", "periode",
                        "nom_usuel", "remediation_optimistic_eur",
                        "remediation_pessimistic_eur", "tonnage_restriction",
                        "tonnage_limit", "aquatic", "prior_repairs"],
        "etat_legend": {v: k for k, v in ETAT_INT.items()},
        "type_legend": {v: k for k, v in TYPE_INT.items()},
        "n":           len(rows),
    }

    with open(MINI_OUT, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "data": rows}, fh,
                  ensure_ascii=False, separators=(",", ":"))

    size_mb = os.path.getsize(MINI_OUT) / 1e6
    print(f"  Done → {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    geojson_mb = os.path.getsize(GEOJSON_OUT) / 1e6
    mini_mb    = os.path.getsize(MINI_OUT)    / 1e6

    print(f"\n  {'File':<30} {'Rows':>8}  {'Size':>8}  {'Per row':>9}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*9}")
    print(f"  {'map_data.geojson':<30} {len(df):>8,}  {geojson_mb:>6.1f} MB"
          f"  {geojson_mb*1e6/len(df):>7.0f} B")
    print(f"  {'map_data_mini.json':<30} {len(df):>8,}  {mini_mb:>6.1f} MB"
          f"  {mini_mb*1e6/len(df):>7.0f} B")
    print(f"\n  Compression ratio (mini/geojson): {mini_mb/geojson_mb:.2f}x")

    print(f"\n  GeoJSON properties: 23 per feature")
    print(f"  Mini columns:       9 per row (id, lat, lon, etat_int, type_int,")
    print(f"                      surface, maint_min, remediation_central, risk_score)")

    print(f"\n  Scenario columns in GeoJSON:")
    print(f"    remediation_optimistic  (×0.70 works)")
    print(f"    remediation_central     (×1.00 works)  ← default")
    print(f"    remediation_pessimistic (×1.30 works)")

    print(f"\n  Coordinate system: WGS84 (EPSG:4326), rounded to 6 dp (~0.1m)")
    print(f"  Mini coordinates:  rounded to 5 dp (~1m)")

    print(f"\n  etat_int encoding: 0=bon_etat  1=altérer  2=significatif  3=majeur")
    print(f"  type_int encoding: 0=pont_tablier  1=pont_voûte  2=mur  3=buse  4=cadre")

    # Spot-check: verify scenario math on one row
    row = df[df["etat_class"] == "defaut_majeur"].iloc[0]
    expected_opt = row["remediation_works"] * 0.70 * 1.10 + (row["study_cost"] - MOE_RATE * row["remediation_works"])
    print(f"\n  Spot-check (one majeur ouvrage):")
    print(f"    remediation_works:      {row['remediation_works']:>10,.2f} €")
    print(f"    study_cost (fixed+MOE): {row['study_cost']:>10,.2f} €")
    print(f"    remediation_central:    {row['remediation_central']:>10,.2f} €")
    print(f"    remediation_optimistic: {row['remediation_optimistic']:>10,.2f} €  (expected {expected_opt:,.2f})")
    print(f"    remediation_pessimistic:{row['remediation_pessimistic']:>10,.2f} €")

    # null/missing type_int check
    unmapped = df["ph1_natured_ge"].isin(["Donnée non accessible"]).sum()
    print(f"\n  Note: {unmapped} 'Donnée non accessible' rows → type_int=-1 in mini format")


# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("MAP EXPORT — SIRO/PNP ouvrage scoring")
    print("=" * 60)

    df = load_and_enrich()
    export_geojson(df)
    export_mini(df)
    print_summary(df)


if __name__ == "__main__":
    main()
