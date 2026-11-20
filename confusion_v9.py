"""
confusion_v9.py - TP / FP / FN / TN for each model on each answer key, same test services (CSN + CPT, quality codes out).

Answer keys come from the runs' own test labels:
  original   test labels of the --base run (GROUP_LABEL)
  v8 pax     test labels of the run named l6pax (LABEL_V6_PAX)
  option C   test labels of the run named l6paxc (LABEL_V6_PAXC)
  agreed     services where all three keys give the same label
Cut-off: per model and key, the score that catches --recall of that key's PA services (default 80% and 90%).

  python confusion_v9.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42 --base base --out v9d_confusion_s42.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

QUALITY = r"^\d{4}F$"


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    return t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_folder; names must include l6pax and l6paxc")
    ap.add_argument("--base", required=True)
    ap.add_argument("--recall", type=float, nargs="+", default=[0.80, 0.90])
    ap.add_argument("--out", default="v9d_confusion.csv")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    S = {k: services(f"{v}/test_preds.parquet") for k, v in runs.items()}
    idx = sorted(set.intersection(*[set(s.index) for s in S.values()]))
    keys = {"original": S[a.base].y.loc[idx], "v8 pax": S["l6pax"].y.loc[idx], "option C": S["l6paxc"].y.loc[idx]}
    agree = keys["original"].eq(keys["v8 pax"]) & keys["original"].eq(keys["option C"])
    print(f"test services: {len(idx):,} | agreed: {int(agree.sum()):,} | PA per key: "
          + ", ".join(f"{k} {int(v.sum()):,}" for k, v in keys.items()) + f", agreed {int(keys['original'][agree].sum()):,}")
    rows = []
    for kname in ["original", "v8 pax", "option C", "agreed"]:
        y = keys["original"][agree] if kname == "agreed" else keys[kname]
        for m in runs:
            s = S[m].s.loc[y.index]
            for r in a.recall:
                cut = float(np.quantile(s[y.eq(1)], 1 - r))
                f = s.ge(cut)
                tp, fp = int((f & y.eq(1)).sum()), int((f & y.eq(0)).sum())
                fn, tn = int((~f & y.eq(1)).sum()), int((~f & y.eq(0)).sum())
                rows.append({"answer key": kname, "model": m, "target recall": f"{r:.0%}", "TP": tp, "FP": fp,
                             "FN": fn, "TN": tn, "precision": round(tp / max(tp + fp, 1), 4),
                             "recall": round(tp / max(tp + fn, 1), 4), "flagged": tp + fp, "cut-off": round(cut, 4)})
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for r in R["target recall"].unique():
        print(f"\n== at {r} recall\n" + R[R["target recall"].eq(r)].drop(columns="target recall").to_string(index=False))
    R.to_csv(a.out, index=False)
    print(f"-> {a.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
