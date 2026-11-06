"""
build_v7.py - v7 dataset = v6 (v4 + quality-code columns) + labels from the business's (Acuity) PA denial codes. New

Source of the code list: "Acuity - Auth Denials CARC Codes" (Anesha Parker / Paula, 22 Oct 2026):
17 CARCs with category = Denial, sub_category = Authorization / Pre-Certification.
  AUTH_ONLY (only row for that code in the sheet):  39 61 62 165 197 198 210 243 P17 W6
  MIXED (code also maps to other sub-categories):    15 16 59 96 109 147 A1

How a line is checked: ALL remit rows of the claim line in the RAW file (v3/...{year}.parquet) are searched;
remark/adjustment codes are in the raw CARC column (one code per remit row). Lines are matched to v6 by
CSN + FINAL_CPT + SERVICEDATE (+ FINAL_MODIFIERS + FILING_ORDER_C when the raw file has them).

New columns (every v6 row and column kept, same order):
  ACUITY_AUTH_CODES   Acuity codes found on any remit row of the line, e.g. "16,197" ('' if none)
  LABEL_V7            full Acuity list: LABEL_V6, set TRUE if ANY of the 17 codes is present      (as requested)
  LABEL_V7_STRICT     auth-only list: LABEL_V6, set TRUE only if an AUTH_ONLY code is present        (recommended check)
  RELABEL_REASON_V7   acuity_auth_only | acuity_mixed | '' (why LABEL_V7 differs from LABEL_V6)
  LABEL_VERSION       'v7.0'
Rules: quality codes (IS_QUALITY_CODE = 1) stay FALSE in both; no label ever goes TRUE -> FALSE in v7
(existing PA codes 284/296/302, M62/N54 keep working as before).

Run:
  python build_v7.py --years 2022 2023 2024 2025 2026 --dry-run          # counts only
  nohup python -u build_v7.py --years 2022 2023 2024 2025 2026 \
     --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7/ai_prior_auth_09282026_v3_{year}_std.parquet" \
     > run_build_v7.log 2>&1 &
"""
import argparse
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

AUTH_ONLY = ["39", "61", "62", "165", "197", "198", "210", "243", "P17", "W6"]
MIXED = ["15", "16", "59", "96", "109", "147", "A1"]
ACUITY = AUTH_ONLY + MIXED
V6 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v6/ai_prior_auth_09282026_v3_{year}_std.parquet"
RAW = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_{year}.parquet"
VERSION = "v7.0"
TRUE_SET = {"TRUE", "1", "1.0"}


def norm_keys(t, keys):
    t = t.copy()
    t["CSN"] = t["CSN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t["FINAL_CPT"].astype(str).str.strip().str.upper()
    t["SERVICEDATE"] = pd.to_datetime(t["SERVICEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "FINAL_MODIFIERS" in keys:
        t["FINAL_MODIFIERS"] = t["FINAL_MODIFIERS"].astype(object).fillna("").astype(str).str.strip().str.upper()
    if "FILING_ORDER_C" in keys:
        t["FILING_ORDER_C"] = pd.to_numeric(t["FILING_ORDER_C"], errors="coerce").fillna(-1).astype(int).astype(str)
    return t


def acuity_hits(year, fs, keys):
    """claim lines (keys) -> comma list of Acuity codes on any of their remit rows (raw file)"""
    p = RAW.format(year=year).replace("s3://", "")
    d = ds.dataset(p, filesystem=fs, format="parquet")
    t = d.to_table(columns=keys + ["CARC"], filter=ds.field("CARC").isin(ACUITY)).to_pandas()
    t["CARC"] = t.CARC.astype(str).str.strip().str.upper()
    t = norm_keys(t, keys)
    h = t.groupby(keys).CARC.agg(lambda s: ",".join(sorted(set(s), key=ACUITY.index))).rename("ACUITY_AUTH_CODES")
    print(f"  [{year}] raw remit rows with an Acuity code: {len(t):,} -> claim lines {len(h):,}")
    return h.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--base-path", default=V6, help="v6 files (with {year})")
    ap.add_argument("--out-path", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="v7_manifest.json")
    a = ap.parse_args()
    if not a.dry_run and ("{year}" not in a.out_path or "std_data_v4" in a.out_path or "std_data_v6" in a.out_path):
        raise SystemExit("--out-path must contain {year} and must not point at v4/v6 (or use --dry-run)")
    fs = pafs.S3FileSystem()
    man = {"version": VERSION, "auth_only": AUTH_ONLY, "mixed": MIXED, "base": a.base_path, "raw": RAW, "years": {}}

    for y in a.years:
        t0 = time.time()
        raw_names = pq.read_schema(RAW.format(year=y).replace("s3://", ""), filesystem=fs).names
        keys = ["CSN", "FINAL_CPT", "SERVICEDATE"] + [c for c in ["FINAL_MODIFIERS", "FILING_ORDER_C"] if c in raw_names]
        hits = acuity_hits(y, fs, keys)
        src = a.base_path.format(year=y).replace("s3://", "")
        d = ds.dataset(src, filesystem=fs, format="parquet")
        if "LABEL_V6" not in d.schema.names:
            raise SystemExit(f"{src}: no LABEL_V6 column - point --base-path at the v6 files")
        writer = out_p = None
        if not a.dry_run:
            out = a.out_path.format(year=y)
            out_fs, out_p = (fs, out.replace("s3://", "")) if out.startswith("s3://") else (pafs.LocalFileSystem(), out)
        c = {"rows": 0, "rows_with_acuity_code": 0, "V6_TRUE": 0, "V7_TRUE": 0, "V7_STRICT_TRUE": 0,
             "to_TRUE_auth_only": 0, "to_TRUE_mixed_only": 0, "quality_kept_FALSE": 0, "matched_keys_check": 0}
        per_code = {k: 0 for k in ACUITY}
        q_code = {k: 0 for k in ACUITY}                 # quality-code lines held FALSE, by Acuity code
        q_orig = {}                                     # ... and their original GROUP_LABEL
        for b in d.to_batches(batch_size=500_000):
            t = b.to_pandas()
            k = norm_keys(t[keys], keys)
            codes = k.merge(hits, on=keys, how="left").ACUITY_AUTH_CODES.fillna("").to_numpy()
            t["ACUITY_AUTH_CODES"] = codes
            v6 = t.LABEL_V6.astype(str).str.strip().str.upper()
            q = t.get("IS_QUALITY_CODE", pd.Series(0, index=t.index)).astype(int).eq(1)
            cs = pd.Series(codes, index=t.index).str.split(",")
            has_auth = cs.map(lambda l: any(x in AUTH_ONLY for x in l))
            has_any = pd.Series(codes, index=t.index).ne("")
            flip_full = has_any & ~v6.isin(TRUE_SET) & ~q
            flip_strict = has_auth & ~v6.isin(TRUE_SET) & ~q
            t["LABEL_V7"] = t.LABEL_V6.astype(object).where(~flip_full, "TRUE").astype(str)
            t["LABEL_V7_STRICT"] = t.LABEL_V6.astype(object).where(~flip_strict, "TRUE").astype(str)
            t["RELABEL_REASON_V7"] = ""
            t.loc[flip_strict, "RELABEL_REASON_V7"] = "acuity_auth_only"
            t.loc[flip_full & ~flip_strict, "RELABEL_REASON_V7"] = "acuity_mixed"
            t["LABEL_VERSION"] = VERSION
            c["rows"] += len(t)
            c["rows_with_acuity_code"] += int(has_any.sum())
            c["V6_TRUE"] += int(v6.isin(TRUE_SET).sum())
            c["V7_TRUE"] += int(t.LABEL_V7.str.upper().isin(TRUE_SET).sum())
            c["V7_STRICT_TRUE"] += int(t.LABEL_V7_STRICT.str.upper().isin(TRUE_SET).sum())
            c["to_TRUE_auth_only"] += int(flip_strict.sum())
            c["to_TRUE_mixed_only"] += int((flip_full & ~flip_strict).sum())
            c["quality_kept_FALSE"] += int((has_any & q & ~v6.isin(TRUE_SET)).sum())
            qh = has_any & q
            for code in ACUITY:
                has_c = cs.map(lambda l: code in l)
                per_code[code] += int((flip_full & has_c).sum())
                q_code[code] += int((qh & has_c).sum())
            g = t.GROUP_LABEL.astype(str).str.strip().str.upper()[qh].value_counts()
            for kk, vv in g.items():
                q_orig[kk] = q_orig.get(kk, 0) + int(vv)
            if not a.dry_run:
                tb = pa.Table.from_pandas(t, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
                writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        if writer is not None:
            writer.close()
            assert pq.read_metadata(out_p, filesystem=out_fs).num_rows == c["rows"], f"{y}: row count mismatch"
        assert c["V7_TRUE"] - c["V6_TRUE"] == c["to_TRUE_auth_only"] + c["to_TRUE_mixed_only"], f"{y}: TRUE count check failed"
        man["years"][y] = c | {"newly_TRUE_by_code": per_code, "quality_held_FALSE_by_code": q_code,
                               "quality_held_original_label": q_orig, "keys": keys}
        print(f"[{y}] rows {c['rows']:,} | with Acuity code {c['rows_with_acuity_code']:,} | TRUE v6 {c['V6_TRUE']:,} -> "
              f"v7 {c['V7_TRUE']:,} (+{c['V7_TRUE'] - c['V6_TRUE']:,}) | v7_strict {c['V7_STRICT_TRUE']:,} "
              f"(+{c['to_TRUE_auth_only']:,}) | quality kept FALSE {c['quality_kept_FALSE']:,} | {time.time() - t0:.0f}s")
        print("      newly TRUE by code:", {k: v for k, v in per_code.items() if v})
        print("      quality codes held FALSE, by code:", {k: v for k, v in q_code.items() if v},
              "| original label:", q_orig)
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(f"manifest -> {a.manifest}")


if __name__ == "__main__":
    main()
