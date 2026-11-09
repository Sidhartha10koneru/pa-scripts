"""
build_v7_1.py - v7.1 = v7 + passenger relabel (a service that is TRUE only because another service's auth
number is on the same claim). Every v7 row and column is kept; new columns are added.

Rule (as measured in v6_label_impact.py, owner rule, run with --rate-basis amb_false --thresholds 0.01 0.02 0.05):
  a line is a PASSENGER when ALL hold
   1. TRUE in GROUP_LABEL through an auth number (auth value present, not "no auth" text)
   2. the same auth value is on another CPT of the same claim (claim = CSN + FILING_ORDER_C)
   3. its CPT is on the "rarely own-PA" list: alone-on-claim PA rate < 2% (AMBIGUOUS counted as not PA,
      >= 100 alone lines, training window)                -> P_list_thr2.csv
   4. (CPT, payor) is not a payor exception (CPT often PA alone at that payor)  -> R2_payor_exceptions.csv
   5. the claim keeps another TRUE line that is neither a quality code nor itself a passenger (= the auth has an
      owner). Claims with no owner left = visit-level / referral auth -> kept TRUE, flagged VISIT_LEVEL_AUTH.
  Quality codes are not passengers (already FALSE in v6/v7).

New columns
  IS_PASSENGER       1 if the rule above holds
  VISIT_LEVEL_AUTH   1 if 1-4 hold but the claim has no owner (kept TRUE; referral decision pending)
  LABEL_V6_PAX       LABEL_V6        with passengers set FALSE
  LABEL_V7_STRICT_PAX LABEL_V7_STRICT with passengers set FALSE, unless the line has its own Acuity code
  LABEL_V7_PAX       LABEL_V7        with passengers set FALSE, unless the line has its own Acuity code
  LABEL_VERSION      'v7.1'
No label ever goes FALSE -> TRUE here.

Inputs: the two CSVs from the owner-rule impact run (default folder v6_impact_owner/).
Run:
  python build_v7_1.py --years 2022 2023 2024 2025 2026 --dry-run
  nohup python -u build_v7_1.py --years 2022 2023 2024 2025 2026 \
     --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7_1/ai_prior_auth_09282026_v3_{year}_std.parquet" \
     > run_build_v7_1.log 2>&1 &
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import pa_base_adapter as pb

V7 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7/ai_prior_auth_09282026_v3_{year}_std.parquet"
VERSION = "v7.1"
BLANK = {"", "NAN", "NONE", "NULL", "<NA>"}
NO_AUTH = r"(NO\s*(REF|REFERRAL|AUTH|PA|PRIOR)\b.*|NOT\s*REQ.*|NAR|NPN|NPAR|N/?A|NONE|0+)"
TRUE_SET = {"TRUE", "1", "1.0"}
NEED = ["CSN", "FILING_ORDER_C", "FINAL_CPT", "PAYOR", "PLAN", "GROUP_LABEL", "LINE_AUTHORIZATION_NUMBER",
        "IS_QUALITY_CODE", "ACUITY_AUTH_CODES"]


def passenger_flags(t, plist, exc):
    """t: one year's lines (row order = file order). Returns IS_PASSENGER, VISIT_LEVEL_AUTH arrays."""
    t = pb.clean_names(t.copy())
    cpt = t.FINAL_CPT.astype(str).str.strip().str.upper()
    claim = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" + \
        pd.to_numeric(t.FILING_ORDER_C, errors="coerce").fillna(1).astype(int).astype(str)
    a = t.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip().str.upper().str.rstrip(",. ")
    auth = a.where(~a.isin(BLANK) & ~a.str.fullmatch(NO_AUTH, na=False), "")
    true = t.GROUP_LABEL.astype(str).str.strip().str.upper().isin(TRUE_SET)
    q = t.IS_QUALITY_CODE.astype(int).eq(1)
    # 2. auth value shared with another CPT of the claim
    f = pd.DataFrame({"claim": claim, "auth": auth, "cpt": cpt})
    n_cpt = f[f.auth != ""].drop_duplicates().groupby(["claim", "auth"]).cpt.nunique()
    shared = pd.Series(list(zip(claim, auth)), index=t.index).map(n_cpt).fillna(0).ge(2)
    # 1-4
    pair = pd.Series(list(zip(cpt, t.PAYOR_CLEAN.astype(str))), index=t.index)
    cand = true & auth.ne("") & shared & cpt.isin(plist) & ~pair.isin(exc) & ~q
    # 5. owner: another TRUE line on the claim that is not quality and not a candidate
    owner_claims = set(claim[true & ~q & ~cand])
    has_owner = claim.isin(owner_claims)
    return (cand & has_owner).to_numpy(), (cand & ~has_owner).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--base-path", default=V7)
    ap.add_argument("--impact-dir", default="v6_impact_owner", help="folder with P_list_thr2.csv, R2_payor_exceptions.csv")
    ap.add_argument("--out-path", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="v7_1_manifest.json")
    a = ap.parse_args()
    if not a.dry_run and ("{year}" not in a.out_path or any(v in a.out_path for v in ["std_data_v4/", "std_data_v6/", "std_data_v7/"])):
        raise SystemExit("--out-path must contain {year} and must not point at v4/v6/v7 (or use --dry-run)")
    plist = set(pd.read_csv(f"{a.impact_dir}/P_list_thr2.csv", dtype=str).FINAL_CPT.str.strip().str.upper())
    e = pd.read_csv(f"{a.impact_dir}/R2_payor_exceptions.csv", dtype=str)
    exc = set(zip(e.FINAL_CPT.str.strip().str.upper(), e.PAYOR_CLEAN.astype(str)))
    print(f"passenger list: {len(plist)} CPTs | payor exceptions: {len(exc)} pairs")
    fs = pafs.S3FileSystem()
    man = {"version": VERSION, "n_list": len(plist), "n_exceptions": len(exc), "base": a.base_path, "years": {}}

    for y in a.years:
        t0 = time.time()
        src = a.base_path.format(year=y).replace("s3://", "")
        names = pq.read_schema(src, filesystem=fs).names
        miss = [c for c in NEED + ["LABEL_V6", "LABEL_V7", "LABEL_V7_STRICT"] if c not in names]
        if miss:
            raise SystemExit(f"{src}: missing columns {miss} - point --base-path at the v7 files")
        d = ds.dataset(src, filesystem=fs, format="parquet")
        t = d.to_table(columns=NEED).to_pandas()
        pax, vla = passenger_flags(t, plist, exc)
        own_code = t.ACUITY_AUTH_CODES.astype(object).fillna("").astype(str).ne("").to_numpy()
        top_cpt = pd.Series(t.FINAL_CPT.astype(str).to_numpy()[pax]).value_counts().head(12).to_dict()
        del t
        writer = out_p = None
        if not a.dry_run:
            out = a.out_path.format(year=y)
            out_fs, out_p = (fs, out.replace("s3://", "")) if out.startswith("s3://") else (pafs.LocalFileSystem(), out)
        c = {"rows": 0, "IS_PASSENGER": int(pax.sum()), "VISIT_LEVEL_AUTH": int(vla.sum())}
        for k in ["LABEL_V6", "LABEL_V6_PAX", "LABEL_V7_STRICT", "LABEL_V7_STRICT_PAX", "LABEL_V7", "LABEL_V7_PAX"]:
            c[f"TRUE_{k}"] = 0
        i = 0
        for b in d.to_batches(batch_size=500_000):
            t = b.to_pandas()
            n = len(t)
            p, v, oc = pax[i:i + n], vla[i:i + n], own_code[i:i + n]
            i += n
            t["IS_PASSENGER"] = p.astype("int8")
            t["VISIT_LEVEL_AUTH"] = v.astype("int8")
            t["LABEL_V6_PAX"] = t.LABEL_V6.astype(object).where(~p, "FALSE").astype(str)
            t["LABEL_V7_STRICT_PAX"] = t.LABEL_V7_STRICT.astype(object).where(~(p & ~oc), "FALSE").astype(str)
            t["LABEL_V7_PAX"] = t.LABEL_V7.astype(object).where(~(p & ~oc), "FALSE").astype(str)
            t["LABEL_VERSION"] = VERSION
            c["rows"] += n
            for k in ["LABEL_V6", "LABEL_V6_PAX", "LABEL_V7_STRICT", "LABEL_V7_STRICT_PAX", "LABEL_V7", "LABEL_V7_PAX"]:
                c[f"TRUE_{k}"] += int(t[k].astype(str).str.strip().str.upper().isin(TRUE_SET).sum())
            if not a.dry_run:
                tb = pa.Table.from_pandas(t, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
                writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        assert i == len(pax), f"{y}: row order mismatch ({i} vs {len(pax)})"
        if writer is not None:
            writer.close()
            assert pq.read_metadata(out_p, filesystem=out_fs).num_rows == c["rows"], f"{y}: row count mismatch"
        assert c["TRUE_LABEL_V6"] - c["TRUE_LABEL_V6_PAX"] <= c["IS_PASSENGER"], f"{y}: passenger count check failed"
        man["years"][y] = c | {"top_passenger_cpts": top_cpt}
        print(f"[{y}] rows {c['rows']:,} | passengers {c['IS_PASSENGER']:,} | visit-level auth kept TRUE "
              f"{c['VISIT_LEVEL_AUTH']:,} | TRUE v6 {c['TRUE_LABEL_V6']:,} -> {c['TRUE_LABEL_V6_PAX']:,} | "
              f"v7_strict {c['TRUE_LABEL_V7_STRICT']:,} -> {c['TRUE_LABEL_V7_STRICT_PAX']:,} | "
              f"v7 {c['TRUE_LABEL_V7']:,} -> {c['TRUE_LABEL_V7_PAX']:,} | {time.time() - t0:.0f}s")
        print("      top passenger CPTs:", top_cpt)
    json.dump(man, open(a.manifest, "w"), indent=1, default=int)
    print(f"manifest -> {a.manifest}")


if __name__ == "__main__":
    main()


#python build_v7_1.py --years 2022 2023 2024 2025 2026 --impact-dir v6_impact_owner --dry-run
#nohup python -u build_v7_1.py --years 2022 2023 2024 2025 2026 --impact-dir v6_impact_owner --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7_1/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_v7_1.log 2>&1 &
