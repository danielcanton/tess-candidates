import numpy as np, lightkurve as lk, warnings, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
P, T0 = 18.05665, 3938.0234
sr = lk.search_lightcurve("TIC 257484419")
want = [(3,"TESS-SPOC"),(30,"SPOC"),(97,"SPOC")]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
i = 0; allx=[]; ally=[]
for s, a in want:
    r = [x for x in sr if x.author[0]==a and int(x.mission[0].split()[-1])==s][0]
    lc = r.download(quality_bitmask="default").remove_nans().normalize()
    lc = lk.LightCurve(time=lc.time, flux=np.asarray(lc.flux.value.filled(np.nan) if hasattr(lc.flux.value,"filled") else lc.flux.value, float)).remove_nans()
    exp = np.median(np.diff(lc.time.value))
    lc = lc.flatten(window_length=int(round(0.75/exp))|1)
    t, f = lc.time.value, lc.flux.value
    for n in range(int(np.ceil((t.min()-T0)/P)), int(np.floor((t.max()-T0)/P))+1):
        tc = T0 + n*P; m = np.abs(t-tc) < 0.35
        if m.sum() < 5 or i >= 5: continue
        x = (t[m]-tc)*24; ax = axes[i]; i += 1
        ax.plot(x, f[m], ".", ms=2, color="0.6")
        if exp < 0.01:
            b = np.arange(-8.4, 8.5, 0.5); idx = np.digitize(x, b)
            ax.plot([(b[k-1]+b[k])/2 for k in set(idx) if 0<k<len(b)], [np.mean(f[m][idx==k]) for k in set(idx) if 0<k<len(b)], "o", ms=4, color="C0")
        ax.axvspan(-1.35, 1.35, color="C1", alpha=0.15); ax.axhline(1-0.0012, ls="--", color="r", lw=0.8)
        ax.set_title(f"S{s} {a}  epoch {n}"); ax.set_xlabel("hours from predicted mid-transit")
        allx.append(x); ally.append(f[m])
axes[0].set_ylim(0.994, 1.004); axes[0].set_ylabel("flattened flux")
fig.suptitle(f"TIC 257484419: every predicted transit at P = {P} d (red dashes: 1200 ppm)")
fig.tight_layout(); fig.savefig("all_transits.png", dpi=110)
print("panels", i)
