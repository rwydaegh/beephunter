"""Diagnostic plots from two simultaneous BeepHunter recordings (stationary test).

Auto-finds the newest Android (phone) + Windows (laptop) JSON in Downloads,
or takes two paths as argv. Produces a multi-panel figure covering frequency
stability, amplitude co-variation, multipath, and an Allan-deviation view that
tells us the achievable Doppler precision vs averaging time.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DL = os.path.expanduser(r"~/Downloads")
C = 343.0

def load(f):
    d = json.load(open(f, encoding="utf-8")); r = d["rows"]
    g = lambda k: np.array([x.get(k) for x in r], dtype=float)
    ua = d["meta"].get("ua", "")
    plat = "phone" if "Android" in ua else ("laptop" if "Windows" in ua else "?")
    return {"plat": plat, "t": g("tw")/1000.0, "f": g("f"), "lvl": g("lvl"),
            "inlvl": g("inlvl"), "cad": g("cad"), "meta": d["meta"]}

def pick_files():
    if len(sys.argv) >= 3:
        return load(sys.argv[1]), load(sys.argv[2])
    files = sorted(glob.glob(os.path.join(DL, "beephunter_*.json")), key=os.path.getmtime, reverse=True)
    recs = [load(f) for f in files]
    phone = next(r for r in recs if r["plat"] == "phone")
    lap = next(r for r in recs if r["plat"] == "laptop")
    return phone, lap

def resample(t, v, grid):
    return np.interp(grid, t - t[0], v)

def smooth(x, k):
    if k < 2: return x.copy()
    ker = np.ones(k)/k
    return np.convolve(x, ker, mode="same")

def adev(y, dt, taus):
    out = []
    for tau in taus:
        m = max(1, int(round(tau/dt)))
        nb = len(y)//m
        if nb < 2: out.append(np.nan); continue
        yb = y[:nb*m].reshape(nb, m).mean(axis=1)
        d = np.diff(yb)
        out.append(np.sqrt(0.5*np.mean(d**2)))
    return np.array(out)

def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    P, L = pick_files()
    t0 = max(P["t"][0], L["t"][0]); t1 = min(P["t"][-1], L["t"][-1])
    dt = 0.2; g = np.arange(0, t1-t0, dt)
    # shift each device's own clock to common origin t0
    pf = np.interp(g, P["t"]-t0, P["f"]);   lf = np.interp(g, L["t"]-t0, L["f"])
    pl = np.interp(g, P["t"]-t0, P["lvl"]);  ll = np.interp(g, L["t"]-t0, L["lvl"])
    pc = np.interp(g, P["t"]-t0, P["cad"]);  lc = np.interp(g, L["t"]-t0, L["cad"])
    ks = int(3.0/dt)
    pls, lls = smooth(pl, ks), smooth(ll, ks)
    # trim convolution edges for honest stats/plots
    e = ks//2
    sl = slice(e, len(g)-e)

    fig, ax = plt.subplots(3, 2, figsize=(14, 11)); fig.suptitle(
        "BeepHunter — stationary side-by-side diagnostic (phone vs laptop)", fontsize=14, fontweight="bold")

    # 1) frequency vs time
    a = ax[0,0]
    a.plot(g, pf, color="#1f77b4", lw=.8, label=f"phone  μ={pf.mean():.2f} σ={pf.std():.2f}")
    a.plot(g, lf, color="#d62728", lw=.8, label=f"laptop μ={lf.mean():.2f} σ={lf.std():.2f}")
    a.set_title("Tracked frequency vs time"); a.set_ylabel("Hz"); a.set_xlabel("s"); a.legend(fontsize=8); a.grid(alpha=.3)

    # 2) frequency difference (phone-laptop): is the wobble common drift or noise?
    a = ax[0,1]; diff = pf-lf
    a.plot(g, diff, color="#555", lw=.8)
    a.axhline(diff.mean(), color="k", ls="--", lw=.8)
    a.set_title(f"Frequency difference  (σ={diff.std():.2f} Hz)  — flat-ish ⇒ source is frequency-stable")
    a.set_ylabel("phone − laptop (Hz)"); a.set_xlabel("s"); a.grid(alpha=.3)

    # 3) level vs time, raw + 3s smooth
    a = ax[1,0]
    a.plot(g, pl, color="#1f77b4", lw=.4, alpha=.4); a.plot(g, pls, color="#1f77b4", lw=1.8, label="phone (3s)")
    a.plot(g, ll, color="#d62728", lw=.4, alpha=.4); a.plot(g, lls, color="#d62728", lw=1.8, label="laptop (3s)")
    a.set_title(f"Band level (stationary!)  swing: phone {np.ptp(pl):.1f} dB, laptop {np.ptp(ll):.1f} dB")
    a.set_ylabel("dB"); a.set_xlabel("s"); a.legend(fontsize=8); a.grid(alpha=.3)

    # 4) scatter slow phone vs laptop level -> common vs multipath
    a = ax[1,1]
    x, y = pls[sl]-pls[sl].mean(), lls[sl]-lls[sl].mean()
    r = np.corrcoef(x, y)[0,1]
    a.scatter(x, y, s=10, alpha=.5, color="#2ca02c")
    lim = max(np.abs(x).max(), np.abs(y).max())*1.05
    a.plot([-lim,lim],[-lim,lim], "k--", lw=.8)
    a.set_title(f"Slow level co-variation  r={r:+.2f}\n(scatter on the diagonal = common source power; spread = local multipath)")
    a.set_xlabel("phone Δlevel (dB)"); a.set_ylabel("laptop Δlevel (dB)"); a.grid(alpha=.3)
    a.set_xlim(-lim,lim); a.set_ylim(-lim,lim)

    # 5) Allan deviation of frequency -> precision vs averaging time + Doppler lines
    a = ax[2,0]
    taus = np.logspace(np.log10(dt), np.log10((t1-t0)/4), 25)
    a.loglog(taus, adev(pf, dt, taus), "o-", color="#1f77b4", ms=3, label="phone freq")
    a.loglog(taus, adev(lf, dt, taus), "s-", color="#d62728", ms=3, label="laptop freq")
    for v, c in [(0.8,"#999"), (1.4,"#444")]:
        a.axhline(3036*v/C, ls=":", color=c, label=f"Doppler @ {v} m/s = {3036*v/C:.1f} Hz")
    a.set_title("Allan deviation — frequency noise vs averaging time τ\n(Doppler must sit ABOVE the noise curve to be measurable)")
    a.set_xlabel("averaging time τ (s)"); a.set_ylabel("ADEV (Hz)"); a.legend(fontsize=8); a.grid(alpha=.3, which="both")

    # 6) histograms of frequency
    a = ax[2,1]
    a.hist(pf, bins=40, alpha=.6, color="#1f77b4", label="phone"); a.hist(lf, bins=40, alpha=.6, color="#d62728", label="laptop")
    a.set_title("Frequency distribution"); a.set_xlabel("Hz"); a.set_ylabel("count"); a.legend(fontsize=8); a.grid(alpha=.3)

    fig.tight_layout(rect=[0,0,1,0.97])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stationary_analysis.png")
    fig.savefig(out, dpi=120)
    print("phone:", P["meta"].get("label"), "| laptop:", L["meta"].get("label"))
    print(f"overlap {t1-t0:.0f}s; phone σf={pf.std():.2f}Hz laptop σf={lf.std():.2f}Hz diffσ={diff.std():.2f}Hz")
    print(f"slow-level corr r={r:+.2f}; level swing phone {np.ptp(pl):.1f}dB laptop {np.ptp(ll):.1f}dB")
    print("ADEV phone @ tau=1s:", f"{adev(pf,dt,[1.0])[0]:.2f} Hz", "@ tau=2s:", f"{adev(pf,dt,[2.0])[0]:.2f} Hz")
    print("saved:", out)

if __name__ == "__main__":
    main()
