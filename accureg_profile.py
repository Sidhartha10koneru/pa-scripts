"""
accureg_profile.py - profile the de-identified AccuReg parquet (one row per HL7 ADT/SIU message, fields already split
into columns <SEG>_<n>; repeating segments IN1 / IN2 / IN3 are one column holding a list of dicts), focused on
payer and plan names.

  python accureg_profile.py                    # all 764k rows (a few minutes)
  python accureg_profile.py --sample 100000    # quick look
Output: printed summary + CSVs in --out (default accureg_profile/ on EFS - keep it in the project area).

Field numbering: <SEG>_0 is the segment name, so IN1_4 = IN1 field 4. MSH is one lower (MSH_8 = MSH-9 message type).
Never printed or saved: PID, NK1, GT1, IN2 (insured person, scrubbed SSNs) and person-level IN1 fields
(insured name / DOB / address / policy and member numbers: IN1-16, 18, 19, 36, 49). Their fill % is shown, not values.

Sections
  1 messages   type (MSH-9), date range (MSH-7), patient class (PV1-2), financial class (PV1-20, code without date)
  2 IN1        coverage, insurances per message, fill % of every IN1 field, top values of plan / company fields,
               company code -> name consistency
  3 IN3        certification / auth status (IN3-2): top values and how many look like auth numbers
  4 overlap    IN1-4 company names vs our claims PAYOR / PLAN (same cleaning as training); exact matches only.
               NOT a mapping - a payer / plan crosswalk must come from a governed source.
"""
import argparse
import ast
import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

DEFAULT = "s3://cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
CLAIMS = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
PERSONAL_IN1 = {16, 18, 19, 36, 49}                 # insured name, DOB, address, policy number, insured id
SHOW_IN1 = [2, 3, 4, 8, 9, 12, 13, 15, 17, 35, 46, 47]
EMPTY = {"", "None", "nan", '""', "NULL"}


def fs_path(p):
    if p.startswith("s3://"):
        import pyarrow.fs as pafs
        return pafs.S3FileSystem(), p.replace("s3://", "")
    return None, p


def clean_name(x):
    x = pd.Series(x).astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True)
    return x.str.replace(r"\s+", " ", regex=True).str.strip()


def as_list(v):
    """list-of-dicts cell -> list of dicts (handles real lists, numpy arrays and their string form)"""
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


def val(x):
    x = "" if x is None else str(x).strip()
    return "" if x in EMPTY else x


def top(s, n=20):
    s = s[s.ne("")]
    return s.value_counts().head(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=DEFAULT)
    ap.add_argument("--out", default="accureg_profile")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--claims-year", type=int, default=2026)
    ap.add_argument("--no-claims", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)

    fs, p = fs_path(a.path)
    d = ds.dataset(p, filesystem=fs, format="parquet")
    names = d.schema.names
    want = [c for c in ["MSH_8", "MSH_6", "PV1_2", "PV1_20", "IN1", "IN3"] if c in names]
    print(f"== file: {a.path} | rows {d.count_rows():,} | columns {len(names)} | reading {want}")
    parts, n = [], 0
    for b in d.to_batches(columns=want, batch_size=100_000):
        parts.append(b.to_pandas()); n += b.num_rows
        if a.sample and n >= a.sample:
            break
    t = pd.concat(parts, ignore_index=True)
    if a.sample:
        t = t.head(a.sample)
    N = len(t)

    # ------------------------------------------------------------------ 1. messages
    print(f"\n== 1. messages ({N:,})")
    if "MSH_8" in t:
        mt = t.MSH_8.map(val).str.replace("^", "_", regex=False)
        print("   message type (MSH-9):\n" + top(mt, 15).to_string())
    if "MSH_6" in t:
        dt = pd.to_datetime(t.MSH_6.map(val).str.slice(0, 8), format="%Y%m%d", errors="coerce")
        print(f"   message date (MSH-7): {dt.min():%Y-%m-%d} .. {dt.max():%Y-%m-%d} ({dt.isna().mean():.1%} unreadable)")
    if "PV1_2" in t:
        print("   patient class (PV1-2):\n" + top(t.PV1_2.map(val), 10).to_string())
    if "PV1_20" in t:
        fc = t.PV1_20.map(val).str.split("^").str[0]
        fc.value_counts().rename("messages").to_csv(f"{a.out}/financial_class.csv")
        print("   financial class (PV1-20, code only):\n" + top(fc, 20).to_string())

    # ------------------------------------------------------------------ 2. IN1
    rows, n_in1 = [], np.zeros(N, int)
    for i, v in enumerate(t["IN1"] if "IN1" in t else []):
        L = as_list(v)
        n_in1[i] = len(L)
        for k, s in enumerate(L):
            r = {"msg": i, "order": k + 1}
            for f in range(1, 54):
                x = val(s.get(f"IN1_{f}"))
                if f in PERSONAL_IN1:
                    x = "x" if x else ""            # keep only filled / not filled
                r[f] = x
            rows.append(r)
    I = pd.DataFrame(rows)
    print(f"\n== 2. IN1 insurance: {(n_in1 > 0).mean():.1%} of messages have one | per message: "
          + ", ".join(f"{k}: {v:,}" for k, v in pd.Series(n_in1).value_counts().sort_index().items()))
    if I.empty:
        print("   no IN1 parsed - check the column format"); os._exit(0)
    fill = pd.Series({f: round(100 * I[f].ne("").mean(), 1) for f in range(1, 54)}, name="filled_%")
    fill.to_csv(f"{a.out}/in1_field_fill.csv", header=True)
    print("   IN1 fields filled (% of IN1 segments, fields > 0%):\n   "
          + ", ".join(f"IN1-{f}: {v}%" for f, v in fill.items() if v > 0))
    labels = {2: "plan id", 3: "company id", 4: "company name", 8: "group number", 9: "group name",
              12: "plan effective date", 13: "plan expiration date", 15: "plan type", 17: "relationship",
              35: "company plan code", 46: "prior insurance plan id", 47: "coverage type"}
    for f in SHOW_IN1:
        s = I[f]
        if s.ne("").any():
            vc = top(s, 500)
            vc.rename("in1_segments").to_csv(f"{a.out}/in1_{f}_{labels[f].replace(' ', '_')}.csv")
            if f not in (12, 13):
                print(f"\n   IN1-{f} {labels[f]} ({s[s.ne('')].nunique():,} distinct):\n" + vc.head(15).to_string())
    pair = I[I[4].ne("")].groupby([3, 4]).size().rename("in1_segments").sort_values(ascending=False)
    pair.index.names = ["company_id", "company_name"]
    pair.to_csv(f"{a.out}/company_id_x_name.csv")
    per_id = I[I[3].ne("")].groupby(3)[4].nunique()
    print(f"\n   company ids: {len(per_id):,}; with more than one company name: {int((per_id > 1).sum()):,}")
    print("   top company id -> name pairs:\n" + pair.head(20).to_string())
    sh = I[4][I[4].ne("")]
    print(f"   company name shape: upper case {100 * (sh == sh.str.upper()).mean():.0f}%, "
          f"has digits {100 * sh.str.contains(r'[0-9]').mean():.0f}%, 'AS OF' {100 * sh.str.upper().str.contains(r'AS OF').mean():.0f}%, "
          f"median length {int(sh.str.len().median())}")

    # ------------------------------------------------------------------ 3. IN3 certification / auth
    if "IN3" in t:
        c = []
        for v in t.IN3:
            for s in as_list(v):
                c.append({f: val(s.get(f"IN3_{f}")) for f in range(1, 29)})
        C = pd.DataFrame(c)
        if not C.empty:
            st = C[2]
            looks_id = st.str.contains(r"\d") & ~st.str.contains(r"\s")
            cat = st.where(~looks_id, "<auth-number-like value>")
            print(f"\n== 3. IN3 certification: {len(C):,} segments | IN3-2 filled {st.ne('').mean():.1%} | "
                  f"auth-number-like {looks_id.mean():.1%}")
            print("   IN3-2 values (numbers grouped):\n" + top(cat, 25).to_string())
            cat.value_counts().rename("in3_segments").to_csv(f"{a.out}/in3_2_certification.csv")
            fill3 = {f: round(100 * C[f].ne("").mean(), 1) for f in range(1, 29)}
            print("   IN3 fields filled: " + ", ".join(f"IN3-{f}: {v}%" for f, v in fill3.items() if v > 0))

    # ------------------------------------------------------------------ 4. overlap with claims (not a mapping)
    if not a.no_claims:
        try:
            cfs, cp = fs_path(CLAIMS.format(year=a.claims_year))
            cl = pd.read_parquet(cp, columns=["PAYOR", "PLAN"], filesystem=cfs).drop_duplicates()
            ref_p, ref_l = set(clean_name(cl.PAYOR)), set(clean_name(cl.PLAN))
            nm = pd.Series(clean_name(I[4]).values)
            nm = nm[nm.ne("")]
            out = []
            for lab, ref in [("claims PAYOR", ref_p), ("claims PLAN", ref_l)]:
                hit = nm.isin(ref)
                out.append({"IN1-4 company name vs": lab, "distinct names": nm.nunique(),
                            "distinct matched": nm[hit].nunique(), "share of IN1 segments matched": f"{hit.mean():.1%}"})
            O = pd.DataFrame(out)
            print(f"\n== 4. exact overlap with claims {a.claims_year} names (cleaned; NOT a mapping)\n" + O.to_string(index=False))
            O.to_csv(f"{a.out}/overlap_with_claims.csv", index=False)
            um = nm[~nm.isin(ref_p)].value_counts()
            um.head(500).rename("in1_segments").to_csv(f"{a.out}/unmatched_company_names.csv")
            print("   most common company names not in claims PAYOR:\n" + um.head(15).to_string())
        except Exception as e:                                     # noqa: BLE001
            print(f"\n== 4. claims names not loaded ({e})")
    print(f"\nwritten: {a.out}/")
    os._exit(0)                                     # pyarrow S3 can hang at interpreter exit


if __name__ == "__main__":
    main()
