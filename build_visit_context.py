"""
build_visit_context.py - visit-context features (backup to patient history): which OTHER services are on the same visit.
Uses only the CPT codes of the visit (CSN) - no labels, no patient history, nothing from other dates.

New columns on every line (quality codes ignored when counting):
  VC_N_OTHER      number of other distinct CPTs on the visit
  VC_HCPCS_J, VC_INFUSION, VC_THERAPY, VC_IMAG, VC_SURG, VC_LAB, VC_EM, VC_CARDIO, VC_MED
                  1 if ANOTHER service on the visit is in that family (the line's own code is not counted)
Writes a copy of the v8 draft files (all columns kept) into --out-path.

  python build_visit_context.py --dry-run
  nohup python -u build_visit_context.py --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v12_visit_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
      > run_build_visit.log 2>&1 &
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from build_pa_history import V8D, QUALITY, family, fs_path, norm

FAMS = ["HCPCS_J", "INFUSION", "THERAPY", "IMAG", "SURG", "LAB", "EM", "CARDIO", "MED"]


def key(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--base-path", default=V8D)
    ap.add_argument("--out-path", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="visit_draft_manifest.json")
    a = ap.parse_args()
    if not a.dry_run:
        bad = ["std_data_v4/", "std_data_v6/", "std_data_v7/", "std_data_v8_draft/", "std_data_v9_draft/", "std_data_v11"]
        if "{year}" not in a.out_path or any(b in a.out_path for b in bad):
            raise SystemExit("--out-path must contain {year} and must not point at an existing data folder")
    t0, man, fam_cache = time.time(), {"base": a.base_path, "years": {}}, {}
    for y in a.years:
        fs, p = fs_path(a.base_path.format(year=y))
        v = pd.read_parquet(p, columns=["CSN", "FINAL_CPT"], filesystem=fs)
        v["CSN"], v["FINAL_CPT"] = key(v.CSN), norm(v.FINAL_CPT)
        v = v[~v.FINAL_CPT.str.fullmatch(QUALITY)].drop_duplicates()
        for c in v.FINAL_CPT.unique():
            if c not in fam_cache:
                fam_cache[c] = family(c)
        v["fam"] = v.FINAL_CPT.map(fam_cache)
        per = pd.crosstab(v.CSN, v.fam).reindex(columns=FAMS, fill_value=0)   # distinct CPTs per family on each visit
        per["N"] = v.groupby("CSN").FINAL_CPT.nunique()
        del v
        dset = ds.dataset(p, filesystem=fs, format="parquet")
        writer = None
        if not a.dry_run:
            out_fs, out_p = fs_path(a.out_path.format(year=y))
        st = {"rows": 0, "with_other": 0, **{f: 0 for f in FAMS}}
        for b in dset.to_batches(batch_size=500_000):
            tb = pa.Table.from_batches([b])
            d = tb.select(["CSN", "FINAL_CPT"]).to_pandas()
            k, cpt = key(d.CSN), norm(d.FINAL_CPT)
            for c in cpt.unique():
                if c not in fam_cache:
                    fam_cache[c] = family(c)
            own = cpt.map(fam_cache)
            q = cpt.str.fullmatch(QUALITY)
            P = per.reindex(k.values).fillna(0).astype(int).reset_index(drop=True)
            new = {"VC_N_OTHER": np.clip(P.N.values - (~q).astype(int).values, 0, None).astype("int16")}
            for f in FAMS:
                other = P[f].values - ((own.values == f) & ~q.values).astype(int)      # remove the line's own code
                new[f"VC_{f}"] = (other > 0).astype("int8")
            st["rows"] += len(d); st["with_other"] += int((new["VC_N_OTHER"] > 0).sum())
            for f in FAMS:
                st[f] += int(new[f"VC_{f}"].sum())
            if a.dry_run:
                continue
            for c, vals in new.items():
                tb = tb.append_column(c, pa.array(vals))
            if writer is None:
                writer = pq.ParquetWriter(out_p, tb.schema, filesystem=out_fs, compression="snappy")
            writer.write_table(tb.cast(writer.schema) if tb.schema != writer.schema else tb)
        if writer is not None:
            writer.close()
        man["years"][str(y)] = st
        print(f"{y}: {st['rows']:,} lines | with other services on the visit {st['with_other'] / max(st['rows'], 1):.1%} | "
              + ", ".join(f"{f} {st[f] / max(st['rows'], 1):.1%}" for f in FAMS)
              + f"{'  [dry run]' if a.dry_run else ''}  ({time.time() - t0:.0f}s)")
    json.dump(man, open(a.manifest, "w"), indent=1)
    print(f"-> {a.manifest}")
    os._exit(0)


if __name__ == "__main__":
    main()
