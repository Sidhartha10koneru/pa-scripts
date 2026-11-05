"""
freeze_test_set.py - freeze ONE official test set so every run is scored on exactly the same services.

Source: the benchmark run's test_preds.parquet (no-filter, 2022-2026, test 2026-05-01 .. 2026-07-31).
The frozen set lists every test LINE (SRC_YEAR + SRC_ROW = row in the v4 file) with its service key and label,
plus IS_QUALITY_CODE. Two scopes are counted:
  all          every test service (as reported by the benchmark)
  official     non-quality services (CPT not ^\\d{4}F$) - the scope used from now on

Writes
  <out>/test_set_lines.parquet     one row per test line (keys, label, quality flag)
  <out>/test_set_services.parquet  one row per service (CSN + FINAL_CPT), label = max of its lines
  <out>/manifest.json              definition + counts + sha256 of both files

Run: python freeze_test_set.py --run exp_nofilter_2022 --name test_v1_2026-05_2026-07 --out frozen/test_v1_2026-05_2026-07
     then copy the folder to S3 (command printed at the end)
"""
import argparse
import hashlib
import json
import os

import pandas as pd

QUALITY = r"^\d{4}F$"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def counts(lines, services):
    pa = lines[lines.y == 1]
    v = services.groupby("CSN").y.max()
    return {"lines": len(lines), "visits": int(lines.CSN.nunique()), "services": len(services),
            "PA_lines": len(pa), "PA_services": int(services.y.sum()), "PA_visits": int(v.sum()),
            "PA_line_rate": round(float(lines.y.mean()), 6), "PA_service_rate": round(float(services.y.mean()), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp_nofilter_2022")
    ap.add_argument("--name", default="test_v1_2026-05_2026-07")
    ap.add_argument("--out", default="frozen/test_v1_2026-05_2026-07")
    ap.add_argument("--s3", default="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/test_sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    keep = [c for c in ["SRC_YEAR", "SRC_ROW", "CSN", "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE",
                        "SERVICEDATE", "SVC_DATE", "FILING_ORDER_C", "IS_PRIMARY", "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN",
                        "GROUP_GUID", "y"] if c in t.columns]
    L = t[keep].copy()
    L["CSN"] = L.CSN.astype(str)
    L["FINAL_CPT"] = L.FINAL_CPT.astype(str).str.strip().str.upper()
    L["IS_QUALITY_CODE"] = L.FINAL_CPT.str.fullmatch(QUALITY).astype("int8")
    S = L.groupby(["CSN", "FINAL_CPT"], observed=True).agg(y=("y", "max"), IS_QUALITY_CODE=("IS_QUALITY_CODE", "first"),
                                                         n_lines=("y", "size")).reset_index()
    lp, sp = f"{a.out}/test_set_lines.parquet", f"{a.out}/test_set_services.parquet"
    L.to_parquet(lp, index=False)
    S.to_parquet(sp, index=False)

    off_L, off_S = L[L.IS_QUALITY_CODE == 0], S[S.IS_QUALITY_CODE == 0]
    m = {"name": a.name, "source_run": a.run, "frozen": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
         "definition": {
             "data": "std v4 (ai_prior_auth_09282026_v3_{year}_std.parquet), years loaded 2022-2026",
             "labels": "GROUP_LABEL TRUE/FALSE; AMBIGUOUS/missing dropped",
             "filing_order": "no filter (primary and secondary kept)",
             "dedup": "CSN+CPT+modifier+SERVICEDATE+FILING_ORDER_C, keep TRUE copy, per year file",
             "window": "lines dated 2026-05-01 <= date < 2026-08-01 AND visit's first service date "
                       "(primary rows; all rows if none) in that window",
             "service": "CSN + FINAL_CPT; label = max of its lines",
             "official_scope": "services whose FINAL_CPT is not a quality code (^\\d{4}F$)"},
         "counts": {"all": counts(L, S), "official_non_quality": counts(off_L, off_S)},
         "line_key": "SRC_YEAR + SRC_ROW = row number in the v4 file" if "SRC_ROW" in L else "CSN+CPT+mod+date+filing",
         "sha256": {"test_set_lines.parquet": sha(lp), "test_set_services.parquet": sha(sp)}}
    json.dump(m, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(m["counts"], indent=1))
    print(f"\nfrozen -> {a.out}/\ncopy to S3:\n  aws s3 cp --recursive {a.out} {a.s3.rstrip('/')}/{a.name}/")


if __name__ == "__main__":
    main()
