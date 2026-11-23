"""
pa_exp_cat.py - runs pa_experiments.py unchanged, except that ID-like columns are always treated as categories.

Why: pa_base_adapter.fit_categories treats any numeric column outside the 5 base features as a number. An NPI or a
department id stored as a number would then be split as "NPI > 1,234,567,890", which means nothing. Columns in
FORCE_CAT (or env PA_FORCE_CAT="A B C") are made categorical instead. Everything else is identical:

  python pa_exp_cat.py baselines <same arguments as pa_experiments.py>
"""
import os
import sys

import pa_base_adapter as pb
import pa_experiments as pe

FORCE_CAT = set(os.environ.get("PA_FORCE_CAT", "SERVICEPROVIDERNPI DEPARTMENT_ID PRODUCT_ID PCPNPI REGION").split())
_orig = pb.fit_categories


def fit_categories(train, cols, min_count=20):
    cats = _orig(train, cols, min_count)
    for c in cols:
        if c in FORCE_CAT and c not in cats:
            vc = train[c].astype(str).value_counts()
            cats[c] = sorted(vc[vc >= min_count].index)        # rare values -> missing, as for the base features
            print(f"   {c}: categorical, {len(cats[c]):,} values with >= {min_count} lines")
    return cats


pb.fit_categories = fit_categories
pe.fit_categories = fit_categories

if __name__ == "__main__":
    pe.main()
