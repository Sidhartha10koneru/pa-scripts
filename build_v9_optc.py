"""
build_v9_optc.py - passenger labels with OPTION C (agreed with the team): a CPT is never relabelled at a payer
where that payer has at least one PA line of the CPT billed ALONE (the only CPT on its claim).

Starts from the v8-draft files (passenger rule already applied) and only UNDOES flips; nothing new is flipped.
Every v8 row and column is kept; new columns are added:
  PAX_PROTECTED      1 if the line was a passenger in v8 but its CPT + payer is protected by option C
  IS_PASSENGER_C     IS_PASSENGER with protected lines removed
  LABEL_V6_PAXC      LABEL_V6_PAX with protected lines back to their LABEL_V6 value (PA)
  LABEL_VERSION_C    'v9.0-draft-optC'

Protected CPT + payer pairs are found on the TRAINING window only (service date before --train-end, default
2026-05-01), so the test months never decide a label rule. "Alone" = only CPT on its claim (claim = CSN +
FILING_ORDER_C), counted on all lines incl. quality codes, as in the impact run. PA = LABEL_V6 TRUE.
Payer = PAYOR cleaned as in training (upper, drop "AS OF ...", squeeze spaces).

Run (dry run first: counts only, writes nothing but the pair list and manifest):
  python build_v9_optc.py --dry-run
  nohup python -u build_v9_optc.py --out-path "s3://<AGREED LOCATION>/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
      > run_build_v9.log 2>&1 &
Agree the output S3 folder with the data owner first. The folder name should contain std_data_v9_draft so
register_run.py tags the runs 'v9d'.
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
VERSION = "v9.0-draft-optC"
TRUE_SET = {"TRUE", "1", "1.0"}
KEYS = ["CSN", "FILING_ORDER_C", "FINAL_CPT", "PAYOR", "SERVICEDATE", "LABEL_V6", "IS_PASSENGER"]


def fs_path(p):
    if p.startswith("s3://"):
        return pafs.S3FileSystem(), p.replace("s3://", "")
    return None, p


def cpt(s):
    return s.astype(str).str.strip().str.upper()


def payer(s):
    s = s.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--base-path", default=V8D)
    ap.add_argument("--out-path", default="")
    ap.add_argument("--train-end", default="2026-05-01", help="protected pairs are found on service dates before this")
    ap.add_argument("--min-alone-pa", type=int, default=1, help="option C: 1 alone PA line protects the pair")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="v9_draft_manifest.json")
    ap.add_argument("--pairs-out", default="v9_protected_pairs.csv")
    a = ap.parse_args()
    if not a.dry_run:
        bad = ["std_data_v4/", "std_data_v6/", "std_data_v7/", "std_data_v8_draft/"]
        if "{year}" not in a.out_path or any(b in a.out_path for b in bad):
            raise SystemExit("--out-path must contain {year} and must not point at an existing v4/v6/v7/v8 folder")
    t0 = time.time()
    train_end = pd.Timestamp(a.train_end)

    # ---------------------------------------------------------------- pass 1: protected CPT + payer pairs
    agg = []
    for y in a.years:
        fs, p = fs_path(a.base_path.format(year=y))
        d = pd.read_parquet(p, columns=KEYS, filesystem=fs)
        claim = d.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" + \
            pd.to_numeric(d.FILING_ORDER_C, errors="coerce").fillna(1).astype(int).astype(str)
        d["cpt"] = cpt(d.FINAL_CPT)
        alone = d.groupby(claim).cpt.transform("nunique").eq(1)
        train = pd.to_datetime(d.SERVICEDATE, errors="coerce").lt(train_end)
        pa_ = d.LABEL_V6.astype(str).str.strip().str.upper().isin(TRUE_SET)
        x = d[alone & train]
        g = pd.DataFrame({"cpt": x.cpt, "payor": payer(x.PAYOR), "alone_pa": pa_[alone & train].astype(int)})
        agg.append(g.groupby(["cpt", "payor"]).agg(alone=("alone_pa", "size"), alone_pa=("alone_pa", "sum")))
        print(f"[pass 1] {y}: {len(d):,} lines, {int((alone & train).sum()):,} alone training lines  ({time.time() - t0:.0f}s)")
        del d, x, g
    P = pd.concat(agg).groupby(level=[0, 1]).sum().reset_index()
    prot = P[P.alone_pa >= a.min_alone_pa].copy()
    prot["alone_pa_rate"] = prot.alone_pa / prot.alone
    prot.sort_values("alone_pa", ascending=False).to_csv(a.pairs_out, index=False)
    key = set(prot.cpt + "|" + prot.payor)
    print(f"[pass 1] protected CPT + payer pairs: {len(key):,} (>= {a.min_alone_pa} alone PA line) -> {a.pairs_out}")

    # ---------------------------------------------------------------- pass 2: new columns
    man = {"version": VERSION, "base": a.base_path, "train_end": a.train_end, "min_alone_pa": a.min_alone_pa,
           "protected_pairs": len(key), "dry_run": a.dry_run, "years": {}}
    for y in a.years:
        fs, p = fs_path(a.base_path.format(year=y))
        dset = ds.dataset(p, filesystem=fs, format="parquet")
        writer = out_fs = None
        if not a.dry_run:
            out_fs, out_p = fs_path(a.out_path.format(year=y))
        c = dict(rows=0, passengers_v8=0, protected=0, passengers_c=0, true_v6=0, true_v6_pax=0, true_v6_paxc=0)
        for b in dset.to_batches(batch_size=500_000):
            tb = pa.Table.from_batches([b])
            d = tb.select(["FINAL_CPT", "PAYOR", "IS_PASSENGER", "LABEL_V6", "LABEL_V6_PAX"]).to_pandas()
            k = cpt(d.FINAL_CPT) + "|" + payer(d.PAYOR)
            pax = pd.to_numeric(d.IS_PASSENGER, errors="coerce").fillna(0).eq(1)
            restore = pax & k.isin(key)
            paxc = d.LABEL_V6_PAX.astype(object).where(~restore, d.LABEL_V6.astype(object))
            c["rows"] += len(d); c["passengers_v8"] += int(pax.sum()); c["protected"] += int(restore.sum())
            c["passengers_c"] += int((pax & ~restore).sum())
            for col, s in [("true_v6", d.LABEL_V6), ("true_v6_pax", d.LABEL_V6_PAX), ("true_v6_paxc", paxc)]:
                c[col] += int(s.astype(str).str.strip().str.upper().isin(TRUE_SET).sum())
            if a.dry_run:
                continue
            tb = tb.append_column("PAX_PROTECTED", pa.array(restore.astype("int8").values))
            tb = tb.append_column("IS_PASSENGER_C", pa.array((pax & ~restore).astype("int8").values))
            tb = tb.append_column("LABEL_V6_PAXC", pa.array(paxc.astype(str).values))
            tb = tb.append_column("LABEL_VERSION_C", pa.array(np.full(len(d), VERSION)))
            if writer is None:
                writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
            writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        if writer is not None:
            writer.close()
        # checks: option C only undoes flips
        assert c["true_v6_paxc"] - c["true_v6_pax"] == c["protected"], f"{y}: PA count check failed"
        assert c["passengers_c"] + c["protected"] == c["passengers_v8"], f"{y}: passenger count check failed"
        man["years"][str(y)] = c
        print(f"[pass 2] {y}: {c['rows']:,} rows | passengers v8 {c['passengers_v8']:,} -> option C {c['passengers_c']:,} "
              f"(kept PA {c['protected']:,}) | PA lines V6 {c['true_v6']:,} | V6_PAX {c['true_v6_pax']:,} | "
              f"V6_PAXC {c['true_v6_paxc']:,}{'  [dry run]' if a.dry_run else ''}  ({time.time() - t0:.0f}s)")
    tot = {k: sum(v[k] for v in man["years"].values()) for k in next(iter(man["years"].values()))}
    man["total"] = tot
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(f"\nTOTAL passengers {tot['passengers_v8']:,} -> {tot['passengers_c']:,} | kept PA by option C {tot['protected']:,} "
          f"| PA lines {tot['true_v6']:,} (V6) / {tot['true_v6_pax']:,} (V6_PAX) / {tot['true_v6_paxc']:,} (V6_PAXC) -> {a.manifest}")
    os._exit(0)                                                # pyarrow S3 can hang at interpreter exit


if __name__ == "__main__":
    main()
