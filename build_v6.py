"""
build_v6.py - build the v6 dataset = std v4 files + new label columns. Nothing in v4 is changed or dropped.

v6.0 (this version): quality codes (CPT Category II, ^\\d{4}F$)
  IS_QUALITY_CODE  1 / 0
  LABEL_V6         'FALSE' for quality codes whatever GROUP_LABEL says; otherwise = GROUP_LABEL (unchanged)
  RELABEL_REASON   'quality_code_cpt2' where LABEL_V6 != GROUP_LABEL (after upper/strip), else ''
  LABEL_VERSION    'v6.0'
v6.1 (later, after the passenger decision) adds the passenger rule on top - same columns, new reason code.

Every row of v4 is written, in the same order, with all original columns. One output file per year.
A manifest (counts per year) is written next to the data so every number can be checked.

Run (dry run first: counts only, no files written):
  python build_v6.py --years 2022 2023 2024 2025 2026 --dry-run
  nohup python -u build_v6.py --years 2022 2023 2024 2025 2026 \
      --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v6/ai_prior_auth_09282026_v3_{year}_std.parquet" \
      > run_build_v6.log 2>&1 &
"""
import argparse
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import pa_base_adapter as pb

QUALITY = r"^\d{4}F$"
VERSION = "v6.0"


def add_columns(t):
    lab = t["GROUP_LABEL"].astype(str).str.strip().str.upper()
    q = t["FINAL_CPT"].astype(str).str.strip().str.upper().str.fullmatch(QUALITY)
    new = t["GROUP_LABEL"].astype(object).where(~q, "FALSE")
    t["IS_QUALITY_CODE"] = q.astype("int8")
    t["LABEL_V6"] = new.astype(str)
    t["RELABEL_REASON"] = ""
    t.loc[q & (lab != "FALSE"), "RELABEL_REASON"] = "quality_code_cpt2"
    t["LABEL_VERSION"] = VERSION
    return t, lab, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out-path", default="", help="output file per year, with {year}; s3://... or local")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--manifest", default="v6_manifest.json")
    a = ap.parse_args()
    if not a.dry_run and "{year}" not in a.out_path:
        raise SystemExit("--out-path must contain {year} (or use --dry-run)")
    if not a.dry_run and "std_data_v4" in a.out_path:
        raise SystemExit("refusing to write into the v4 folder")
    fs = pafs.S3FileSystem()
    manifest = {"version": VERSION, "rule": "quality codes ^\\d{4}F$ -> LABEL_V6 = FALSE", "years": {}}

    for y in a.years:
        t0 = time.time()
        src = pb.BASE_PATH.format(year=y).replace("s3://", "")
        d = ds.dataset(src, filesystem=fs, format="parquet")
        writer, out = None, None
        if not a.dry_run:
            out = a.out_path.format(year=y)
            out_fs, out_p = (fs, out.replace("s3://", "")) if out.startswith("s3://") else (pafs.LocalFileSystem(), out)
        c = {"rows": 0, "quality_rows": 0, "changed": 0, "TRUE_to_FALSE": 0, "AMBIGUOUS_to_FALSE": 0,
             "GROUP_TRUE": 0, "V6_TRUE": 0}
        for b in d.to_batches(batch_size=500_000):
            t = b.to_pandas()
            t, lab, q = add_columns(t)
            v6 = t.LABEL_V6.str.strip().str.upper()
            c["rows"] += len(t)
            c["quality_rows"] += int(q.sum())
            c["changed"] += int((t.RELABEL_REASON != "").sum())
            c["TRUE_to_FALSE"] += int((q & lab.isin(["TRUE", "1", "1.0"])).sum())
            c["AMBIGUOUS_to_FALSE"] += int((q & ~lab.isin(["TRUE", "1", "1.0", "FALSE", "0", "0.0"])).sum())
            c["GROUP_TRUE"] += int(lab.isin(["TRUE", "1", "1.0"]).sum())
            c["V6_TRUE"] += int(v6.isin(["TRUE", "1", "1.0"]).sum())
            if not a.dry_run:
                tb = pa.Table.from_pandas(t, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
                writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        if writer is not None:
            writer.close()
        assert c["GROUP_TRUE"] - c["V6_TRUE"] == c["TRUE_to_FALSE"], "TRUE count check failed"
        manifest["years"][y] = c | {"source": src, "output": out}
        print(f"[{y}] rows {c['rows']:,} | quality {c['quality_rows']:,} | TRUE->FALSE {c['TRUE_to_FALSE']:,} | "
              f"AMBIGUOUS->FALSE {c['AMBIGUOUS_to_FALSE']:,} | TRUE {c['GROUP_TRUE']:,} -> {c['V6_TRUE']:,} | "
              f"{'dry run' if a.dry_run else out} ({time.time() - t0:.0f}s)")
        if not a.dry_run:                                   # verify the written file
            w = pq.read_metadata(out_p, filesystem=out_fs)
            assert w.num_rows == c["rows"], f"{y}: written {w.num_rows} rows, expected {c['rows']}"
    json.dump(manifest, open(a.manifest, "w"), indent=1)
    print(f"manifest -> {a.manifest}")


if __name__ == "__main__":
    main()
