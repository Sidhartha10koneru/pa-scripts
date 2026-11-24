"""
build_pa_history.py - PA-history features: did this patient have PA for the same CPT / CPT family recently?

Aimed at the grey-zone false alarms (therapy, infusion, oncology drugs, imaging), where PA usually covers a SERIES of
visits: once a patient has an authorised course, later services of the same kind are likely covered by it.

Reads the v8 draft files (all years, so history crosses year boundaries) and writes a copy of every file with new columns:
  PAH_CPT_90D    number of earlier days (within 90 days before the service) on which the patient had a PA service, same CPT
  PAH_FAM_90D    same, any CPT in the same family (therapy, infusion, drugs, imaging, E/M, ...)
  PAH_ANY_90D    same, any CPT
  PAH_FAM_365D   same family, 365 days
Rules: only days STRICTLY BEFORE the service date, minus --lag-days (default 0) - the service's own label is never used.
A day counts once per patient+CPT (or family). PA = LABEL_V6_PAX TRUE (quality codes excluded).
--lag-days 30 mimics a delay before past PA outcomes are known in production (test it if the API cannot see recent auths).

Patient id (PAT_ID) is used only to link a patient's own history; it is not a model feature and is not written out new.

  python build_pa_history.py --dry-run                     # counts + feature fill on a sample, writes nothing
  nohup python -u build_pa_history.py --out-path "s3://<AGREED>/std_data_v11_hist_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
      > run_build_hist.log 2>&1 &
Agree the output folder with the data owner first.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

V8D = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
TRUE_SET = {"TRUE", "1", "1.0"}
QUALITY = r"^\d{4}F$"
EPOCH = np.datetime64("2000-01-01")
STEP = 100_000                                   # composite key = key * STEP + day (days since 2000 stay < STEP)


def fs_path(p):
    return (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (None, p)


def family(c):
    c = str(c).strip().upper()
    if not c:
        return "unknown"
    if c[0].isalpha():
        return "HCPCS_" + c[0]
    if c.endswith("T"):
        return "CAT3"
    if not c[:5].isdigit():
        return "other"
    n = int(c[:5])
    for lo, hi, name in [(99202, 99499, "EM"), (100, 1999, "ANES"), (10004, 69990, "SURG"), (70010, 79999, "IMAG"),
                         (80047, 89398, "LAB"), (96360, 96549, "INFUSION"), (97010, 97799, "THERAPY"),
                         (92920, 93799, "CARDIO"), (90281, 99199, "MED")]:
        if lo <= n <= hi:
            return name
    return "other"


def days(s):
    d = pd.to_datetime(s, errors="coerce").values.astype("datetime64[D]")
    out = (d - EPOCH).astype("int64")
    return np.where(np.isnat(d), -1, out)


def norm(s):
    return s.astype(str).str.strip().str.upper()


class History:
    """PA events per key, as a sorted composite array; counts events in [day - hi, day - lo] per query."""

    def __init__(self, keys, day):
        k = pd.Series(keys).astype(str).values
        ev = pd.DataFrame({"k": k, "d": day}).drop_duplicates()
        self.cats = pd.Index(pd.unique(ev.k))
        code = self.cats.get_indexer(ev.k)
        self.E = np.sort(code.astype("int64") * STEP + ev.d.values.astype("int64"))

    def count(self, keys, day, lo, hi):
        code = self.cats.get_indexer(pd.Series(keys).astype(str).values).astype("int64")
        ok = (code >= 0) & (day >= 0)
        base = code * STEP
        a = np.searchsorted(self.E, base + (day - hi), side="left")
        b = np.searchsorted(self.E, base + (day - lo), side="right")
        return np.where(ok, b - a, 0).astype("int16")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--base-path", default=V8D)
    ap.add_argument("--out-path", default="")
    ap.add_argument("--label-col", default="LABEL_V6_PAX")
    ap.add_argument("--patient-col", default="PAT_ID")
    ap.add_argument("--lag-days", type=int, default=0, help="ignore history from the last N days before the service")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="hist_draft_manifest.json")
    a = ap.parse_args()
    if not a.dry_run:
        bad = ["std_data_v4/", "std_data_v6/", "std_data_v7/", "std_data_v8_draft/", "std_data_v9_draft/"]
        if "{year}" not in a.out_path or any(b in a.out_path for b in bad):
            raise SystemExit("--out-path must contain {year} and must not point at an existing data folder")
    t0 = time.time()
    lo = 1 + a.lag_days

    # ------------------------------------------------------------ pass 1: PA events (patient, CPT / family, day)
    parts = []
    for y in a.years:
        fs, p = fs_path(a.base_path.format(year=y))
        d = pd.read_parquet(p, columns=[a.patient_col, "FINAL_CPT", "SERVICEDATE", a.label_col], filesystem=fs)
        cpt = norm(d.FINAL_CPT)
        m = norm(d[a.label_col]).isin(TRUE_SET) & ~cpt.str.fullmatch(QUALITY)
        e = pd.DataFrame({"pat": norm(d[a.patient_col][m]), "cpt": cpt[m], "day": days(d.SERVICEDATE[m])})
        e = e[(e.day >= 0) & ~e.pat.isin(["", "NAN", "NONE"])].drop_duplicates()
        parts.append(e)
        print(f"[pass 1] {y}: {len(d):,} lines, {len(e):,} PA patient-CPT-days  ({time.time() - t0:.0f}s)")
        del d
    E = pd.concat(parts, ignore_index=True).drop_duplicates()
    E["fam"] = E.cpt.map({c: family(c) for c in E.cpt.unique()})
    H_cpt = History(E.pat + "|" + E.cpt, E.day.values)
    H_fam = History(E.pat + "|" + E.fam, E.day.values)
    H_any = History(E.pat, E.day.values)
    print(f"[pass 1] PA events {len(E):,} | patients with any PA {E.pat.nunique():,}")

    # ------------------------------------------------------------ pass 2: features for every line
    man = {"base": a.base_path, "label_col": a.label_col, "lag_days": a.lag_days, "dry_run": a.dry_run, "years": {}}
    fam_cache = {}
    for y in a.years:
        fs, p = fs_path(a.base_path.format(year=y))
        dset = ds.dataset(p, filesystem=fs, format="parquet")
        writer = None
        if not a.dry_run:
            out_fs, out_p = fs_path(a.out_path.format(year=y))
        st = {"rows": 0, "cpt90>0": 0, "fam90>0": 0, "any90>0": 0, "fam365>0": 0}
        for b in dset.to_batches(batch_size=500_000):
            tb = pa.Table.from_batches([b])
            d = tb.select([a.patient_col, "FINAL_CPT", "SERVICEDATE"]).to_pandas()
            pat, cpt, day = norm(d[a.patient_col]), norm(d.FINAL_CPT), days(d.SERVICEDATE)
            for c in cpt.unique():
                if c not in fam_cache:
                    fam_cache[c] = family(c)
            fam = cpt.map(fam_cache)
            f = {"PAH_CPT_90D": H_cpt.count(pat + "|" + cpt, day, lo, 90),
                 "PAH_FAM_90D": H_fam.count(pat + "|" + fam, day, lo, 90),
                 "PAH_ANY_90D": H_any.count(pat, day, lo, 90),
                 "PAH_FAM_365D": H_fam.count(pat + "|" + fam, day, lo, 365)}
            st["rows"] += len(d)
            for k, col in [("cpt90>0", "PAH_CPT_90D"), ("fam90>0", "PAH_FAM_90D"), ("any90>0", "PAH_ANY_90D"),
                           ("fam365>0", "PAH_FAM_365D")]:
                st[k] += int((f[col] > 0).sum())
            if a.dry_run:
                continue
            for col, v in f.items():
                tb = tb.append_column(col, pa.array(v))
            if writer is None:
                writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
            writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        if writer is not None:
            writer.close()
        man["years"][str(y)] = st
        print(f"[pass 2] {y}: {st['rows']:,} lines | share with history > 0: CPT 90d {st['cpt90>0'] / max(st['rows'], 1):.1%}, "
              f"family 90d {st['fam90>0'] / max(st['rows'], 1):.1%}, any 90d {st['any90>0'] / max(st['rows'], 1):.1%}, "
              f"family 365d {st['fam365>0'] / max(st['rows'], 1):.1%}{'  [dry run]' if a.dry_run else ''}  ({time.time() - t0:.0f}s)")
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(f"-> {a.manifest}")
    os._exit(0)                                       # pyarrow S3 can hang at interpreter exit


if __name__ == "__main__":
    main()
