"""
accureg_peek.py - look at the values of any AccuReg column (or one field inside a list column like IN1, DG1, PR1).

  python accureg_peek.py --list                 # all columns, grouped by segment, with fill %
  python accureg_peek.py PV1_10                 # one plain column: fill %, distinct, top values
  python accureg_peek.py PV2_3 PV1_18 --top 30  # several columns
  python accureg_peek.py DG1                    # list column: which fields are filled
  python accureg_peek.py DG1 --field 3          # one field inside it (DG1-3 = diagnosis code)
  python accureg_peek.py IN1 --field 4 --top 50
  python accureg_peek.py PV1_3 --rows 764100    # all rows (default: first 100,000)

Patient-level columns are blocked (PID, PD1, NK1, GT1, IN2, doctor names, insured-person IN1 fields): only fill % is shown.
Output stays on screen; nothing is written.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

PATH = "cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
BLOCK_SEG = ("PID", "PD1", "NK1", "GT1", "IN2")
BLOCK_COL = {"PV1_7", "PV1_8", "PV1_9", "PV1_17", "PD1_4"}
BLOCK_FIELD = {"IN1": {16, 18, 19, 36, 49}}          # insured name, DOB, address, policy no., insured id
EMPTY = {"", "None", "nan", '""', "NULL", "[]"}


def val(x):
    x = "" if x is None else str(x).strip()
    return "" if x in EMPTY else x


def as_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        if not v.strip().startswith("["):
            return []
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    if isinstance(v, dict):
        return [v]
    return [x for x in list(v) if isinstance(x, dict)]


def blocked(col, field=None):
    seg = col.split("_")[0]
    return seg.startswith(BLOCK_SEG) or col in BLOCK_COL or (field is not None and field in BLOCK_FIELD.get(col, set()))


def show(s, name, top):
    s = s.map(val)
    f = s[s.ne("")]
    print(f"\n== {name}: filled {len(f) / max(len(s), 1):.1%} of {len(s):,} rows | distinct {f.nunique():,}")
    if len(f):
        print(f.value_counts().head(top).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("columns", nargs="*")
    ap.add_argument("--field", type=int, default=None, help="field number inside a list column (e.g. DG1 --field 3)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    d = ds.dataset(PATH, filesystem=pafs.S3FileSystem(), format="parquet")
    names = d.schema.names

    if a.list or not a.columns:
        t = d.head(min(a.rows, 20_000)).to_pandas()
        fill = {c: t[c].map(val).ne("").mean() for c in names}
        for seg in sorted({c.split("_")[0] for c in names}):
            cols = [c for c in names if c.split("_")[0] == seg and fill[c] > 0]
            if cols:
                print(f"{seg:4s}: " + ", ".join(f"{c} {fill[c]:.0%}" for c in cols))
        print(f"\n({sum(v == 0 for v in fill.values())} empty columns not listed; fill % from the first {len(t):,} rows)")
        os._exit(0)

    missing = [c for c in a.columns if c not in names]
    if missing:
        print("not in file:", missing, "| try --list"); os._exit(0)
    t = d.head(a.rows, columns=a.columns).to_pandas()
    for c in a.columns:
        first = t[c].dropna().head(200)
        is_list = len(first) and first.map(lambda v: len(as_list(v)) > 0).mean() > 0.5
        if not is_list:
            if blocked(c):
                print(f"\n== {c}: blocked (patient-level) | filled {t[c].map(val).ne('').mean():.1%}")
            else:
                show(t[c], c, a.top)
            continue
        recs = [s for v in t[c] for s in as_list(v)]
        n_per = t[c].map(lambda v: len(as_list(v)))
        print(f"\n== {c} (list): {len(recs):,} entries in {int((n_per > 0).sum()):,} of {len(t):,} rows "
              f"| per row: " + ", ".join(f"{k}: {v:,}" for k, v in n_per.value_counts().sort_index().head(6).items()))
        R = pd.DataFrame(recs)
        if a.field is None:
            fields = sorted([k for k in R.columns if k.startswith(f"{c}_")], key=lambda k: int(k.split("_")[-1]))
            print("   fields filled: " + ", ".join(f"{k.split('_')[-1]}: {R[k].map(val).ne('').mean():.0%}"
                                                 for k in fields if R[k].map(val).ne("").any()))
            print(f"   pick one with: python accureg_peek.py {c} --field <number>")
        elif blocked(c, a.field):
            print(f"   {c}-{a.field}: blocked (patient-level) | filled {R.get(f'{c}_{a.field}', pd.Series(dtype=str)).map(val).ne('').mean():.1%}")
        elif f"{c}_{a.field}" not in R:
            print(f"   {c}-{a.field} not present")
        else:
            show(R[f"{c}_{a.field}"], f"{c}-{a.field}", a.top)
    os._exit(0)                                    # pyarrow S3 can hang at interpreter exit


if __name__ == "__main__":
    main()
