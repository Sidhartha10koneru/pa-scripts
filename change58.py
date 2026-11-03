v_pt = pd.concat([ds.dataset(pb.BASE_PATH.format(year=2026).replace("s3://", ""), filesystem=fs, format="parquet")
                  .to_table(filter=ds.field("FINAL_CPT").isin(["97110", "97112", "97140"])).to_pandas()])
v_pt = pb.clean_names(v_pt.loc[:, ~v_pt.columns.duplicated()])
v_pt = v_pt[v_pt.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH / GHI | NYCE PPO PLAN"].copy()
v_pt["m"] = pd.to_datetime(v_pt.SERVICEDATE).dt.to_period("M")
v_pt["L"] = v_pt.GROUP_LABEL.astype(str).str.upper()
print(pd.crosstab(v_pt.m, v_pt.L))                                                    # incl. AMBIGUOUS per month
tt = v_pt[v_pt.L == "TRUE"]
print(pd.crosstab(tt.m, [tt.LEVEL1_LABEL.astype(str), tt.LEVEL2_LABEL.astype(str)]))  # auth number vs denial code
print(tt.LINE_AUTHORIZATION_NUMBER.astype(str).str[:4].value_counts().head(10))       # auth number format

PA_CARC = {"15", "39", "197", "198", "210", "284", "296", "302"}
tt = v_pt[v_pt.L == "TRUE"].copy()
au = tt.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip().str.upper()
carc = tt.CARC.astype(object).fillna("").astype(str).str.upper()
rarc = tt.RARC.astype(object).fillna("").astype(str).str.upper()
tt["has_auth"] = (au != "") & ~au.str.match(r"NO\s*(REF|AUTH)|NOT\s*REQ|NAR|N/?A|NONE")
tt["pa_carc"] = carc.str.findall(r"\d+").map(lambda c: bool(PA_CARC & set(c)))
tt["pa_rarc"] = rarc.str.contains(r"\b(?:M62|N54)\b") | carc.str.contains(r"\b(?:M62|N54)\b")
tt["source"] = np.select([tt.has_auth, tt.pa_carc, tt.pa_rarc], ["auth number", "PA CARC", "PA RARC"],
                         "none on first row (another remit row)")
print(pd.crosstab(tt.m, tt.source, margins=True))
print(tt[tt.pa_carc].CARC.value_counts().head(), tt[tt.has_auth].LINE_AUTHORIZATION_NUMBER.astype(str).head(10).tolist())
python cell_label_check.py --cpt 97112 --modifier GP --pos 11 --plan "EMBLEM HEALTH / GHI | NYCE PPO PLAN" --years 2026 --out cell_check_emblem_ghi_97112
