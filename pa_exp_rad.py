"""
pa_exp_rad.py - pa_exp_cat.py (ID columns categorical) trained and tested on ONE CPT range only (default radiology).

Everything else is unchanged: same loading, cleaning, labels (PA_LABEL_COL), split, features, early stopping.
The only change: right after loading, lines whose CPT is outside --cpt range are dropped (env PA_CPT_RANGE="70010 79999").
The drop is logged in removal_funnel.csv like every other step.

  PA_CPT_RANGE="70010 79999" PA_LABEL_COL=LABEL_V7_STRICT python pa_exp_rad.py baselines <same arguments as pa_experiments.py>
"""
import os

import pandas as pd

import pa_base_adapter as pb
import pa_exp_cat  # noqa: F401  (applies the categorical patch)
import pa_experiments as pe

LO, HI = (int(v) for v in os.environ.get("PA_CPT_RANGE", "70010 79999").split())
_load = pb.load


def load(*args, **kwargs):
    d = _load(*args, **kwargs)
    c = d[pb.CPT].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    n = pd.to_numeric(c.where(c.str.fullmatch(r"\d{5}")), errors="coerce")
    keep = n.between(LO, HI).values
    print(f"  CPT range {LO}-{HI}: keeping {keep.sum():,} of {len(d):,} lines ({keep.mean():.1%})")
    d = d[keep].reset_index(drop=True)
    if hasattr(pb, "log_step"):
        pb.log_step("all", f"4b. CPT outside {LO}-{HI} dropped (single-family model)", d)
    return d


pb.load = load

if __name__ == "__main__":
    pe.main()
