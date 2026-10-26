e = e.sort_values("p", ascending=False)
e["caught"] = e.y.cumsum() / e.y.sum(); e["reviewed"] = np.arange(1, len(e)+1) / len(e)
for target in [0.80, 0.90, 0.95, 0.98]:
    r = e[e.caught >= target].iloc[0]
    print(f"catch {target:.0%} of PA -> review {r.reviewed:.1%} of visits (cut-off {r.p:.3f})")
