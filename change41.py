import pandas as pd, pyarrow.fs as pafs
# a) M62/N54 in the CARC column, and is RARC ever populated?
print("CARC M62/N54 rows:", cs.str.upper().str.contains(r"\b(?:M62|N54)\b", regex=True).sum(),
      "| RARC non-blank share:", (~blank(rs)).mean().round(3))

# b) final dataset: how much is AMBIGUOUS, and is it concentrated in groups that have PA?
s = pd.read_parquet("cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet",
                    filesystem=pafs.S3FileSystem(), columns=["FINAL_CPT", "PAYOR", "PLAN", "GROUP_LABEL"])
s["GROUP_LABEL"] = s.GROUP_LABEL.astype(str)
print(s.GROUP_LABEL.value_counts(normalize=True).round(3))
g = s.groupby(["FINAL_CPT", "PAYOR", "PLAN"]).GROUP_LABEL.value_counts().unstack(fill_value=0)
for c in ["TRUE", "FALSE", "AMBIGUOUS"]:
    g[c] = g.get(c, 0)
g["n"] = g[["TRUE", "FALSE", "AMBIGUOUS"]].sum(1)
g["has_pa"] = g.TRUE > 0
t = g.groupby("has_pa")[["TRUE", "FALSE", "AMBIGUOUS", "n"]].sum()
t["amb_share"] = t.AMBIGUOUS / t.n
t["pa_rate_labelled"] = t.TRUE / (t.TRUE + t.FALSE)
t["pa_rate_if_amb_false"] = t.TRUE / t.n
print(t.round(3))

# c) 99214 by plan: PA rate now vs if AMBIGUOUS were FALSE
q = g.loc["99214"].sort_values("TRUE", ascending=False).head(12).copy()
q["pa_rate_labelled"] = q.TRUE / (q.TRUE + q.FALSE)
q["pa_rate_if_amb_false"] = q.TRUE / q.n
print(q[["TRUE", "FALSE", "AMBIGUOUS", "pa_rate_labelled", "pa_rate_if_amb_false"]].round(3).to_string())
