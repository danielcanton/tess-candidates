import warnings, numpy as np, batman, lightkurve as lk
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
P0, T00 = 18.05665, 3938.0234
U = [0.50, 0.17]  # K dwarf, TESS band (Claret 2017, ~4200 K, logg 4.6)
sr = lk.search_lightcurve("TIC 257484419")
segs = []  # (sector, epoch, t, f, e, cadence_days)
for s, a in ((3, "TESS-SPOC"), (30, "SPOC"), (97, "SPOC")):
    r = [x for x in sr if x.author[0] == a and int(x.mission[0].split()[-1]) == s][0]
    lc = r.download(quality_bitmask="default")
    t = np.asarray(lc.time.value, float)
    f = lc.flux.value; f = np.asarray(f.filled(np.nan) if hasattr(f, "filled") else f, float)
    e = lc.flux_err.value; e = np.asarray(e.filled(np.nan) if hasattr(e, "filled") else e, float)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e); t, f, e = t[ok], f[ok], e[ok]
    cad = np.median(np.diff(t))
    for n in range(int(np.ceil((t.min() - T00) / P0)), int(np.floor((t.max() - T00) / P0)) + 1):
        tc = T00 + n * P0; m = np.abs(t - tc) < 0.6
        if m.sum() < 10: continue
        tt, ff, ee = t[m], f[m], e[m]
        oot = np.abs(tt - tc) > 0.12
        print(f"s{s} epoch {n}: data from {(tt.min()-tc)*24:+.1f} h to {(tt.max()-tc)*24:+.1f} h, {m.sum()} pts, out-of-transit before {np.sum(oot & (tt<tc))} after {np.sum(oot & (tt>tc))}")
        # local quadratic on out-of-transit
        cf = np.polyfit(tt[oot] - tc, ff[oot], 2); base = np.polyval(cf, tt - tc)
        fn = ff / base; en = ee / base
        en = np.full_like(en, np.std(fn[oot]))  # empirical scatter
        segs.append((s, n, tt, fn, en, cad))

def model(p, t, cad):
    t0, per, rp, b, ars = p
    pm = batman.TransitParams(); pm.t0, pm.per, pm.rp, pm.a = t0, per, rp, ars
    pm.inc = np.degrees(np.arccos(min(b / ars, 1))); pm.ecc, pm.w, pm.u, pm.limb_dark = 0.0, 90.0, U, "quadratic"
    sup = 15 if cad > 0.01 else 1
    return batman.TransitModel(pm, t, supersample_factor=sup, exp_time=cad).light_curve(pm)

T = np.concatenate([x[2] for x in segs]); F = np.concatenate([x[3] for x in segs]); E = np.concatenate([x[4] for x in segs])
C = np.concatenate([np.full(len(x[2]), x[5]) for x in segs])
def resid(p, fix_ars=None):
    q = list(p) if fix_ars is None else list(p) + [fix_ars]
    out = np.empty_like(F)
    for cad in np.unique(C):
        m = C == cad; out[m] = model(q, T[m], cad)
    return (F - out) / E

G = 6.674e-8
def ars_from_rho(rho, per): return (G * rho * (per * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)

print("\n== free fit (t0, P, Rp/R*, b, a/R*)")
best = None
for b0 in (0.1, 0.5, 0.8, 0.95):
    r = least_squares(resid, [T00, P0, 0.036, b0, 35], bounds=([T00 - 0.05, P0 - 0.003, 0.005, 0, 3], [T00 + 0.05, P0 + 0.003, 0.4, 1.4, 200]))
    if best is None or r.cost < best.cost: best = r
J = best.jac; cov = np.linalg.pinv(J.T @ J); err = np.sqrt(np.diag(cov))
names = ["t0", "P", "rp", "b", "a/R*"]
for k, v, s_ in zip(names, best.x, err): print(f"  {k} = {v:.6f} +- {s_:.6f}")
chi_free = 2 * best.cost
rho_fit = 3 * np.pi * best.x[4] ** 3 / (G * (best.x[1] * 86400) ** 2)
print(f"  chi2 {chi_free:.1f} for {len(F)} pts; implied stellar density {rho_fit:.2f} g/cm3; Rp = {best.x[2]*0.65*109.1:.2f} R_earth (R* 0.65)")

print("\n== b profile with a/R* free (delta chi2 vs best)")
for bfix in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1):
    f2 = lambda p: resid([p[0], p[1], p[2], bfix, p[3]])
    r = least_squares(f2, [best.x[0], best.x[1], best.x[2], best.x[4]], bounds=([T00 - 0.05, P0 - 0.003, 0.005, 3], [T00 + 0.05, P0 + 0.003, 0.4, 200]))
    print(f"  b={bfix:.1f}: dchi2 {2*r.cost - chi_free:+.1f}, rp {r.x[2]:.4f}, a/R* {r.x[3]:.1f}, rho {3*np.pi*r.x[3]**3/(G*(r.x[1]*86400)**2):.2f}")

for rho in (2.5, 3.3):
    ars = ars_from_rho(rho, P0)
    f3 = lambda p: resid([p[0], p[1], p[2], p[3]], fix_ars=ars)
    r = least_squares(f3, [best.x[0], best.x[1], best.x[2], 0.5], bounds=([T00 - 0.05, P0 - 0.003, 0.005, 0], [T00 + 0.05, P0 + 0.003, 0.4, 1.4]))
    print(f"\n== density fixed at {rho} g/cm3 (a/R* {ars:.1f}): b {r.x[3]:.2f}, rp {r.x[2]:.4f} ({r.x[2]*0.65*109.1:.2f} R_earth), dchi2 vs free {2*r.cost - chi_free:+.1f}")

print("\n== per-transit depth scale (best shape, local quadratic baseline)")
for s, n, tt, fn, en, cad in segs:
    mdl = model(list(best.x), tt, cad) - 1
    A = mdl / en; y = (fn - 1) / en
    k = (A @ y) / (A @ A); sk = 1 / np.sqrt(A @ A)
    print(f"  s{s} epoch {n}: scale {k:.2f} +- {sk:.2f}")
