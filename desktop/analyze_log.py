"""Analyze a BeepHunter CSV log: where/when was the signal strongest?

Usage:
    py -3.14 analyze_log.py [path-to-csv]   (defaults to newest log)
"""
import csv
import glob
import sys
from datetime import datetime

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    t = np.array([datetime.fromisoformat(r["iso_time"]).timestamp() for r in rows])
    t -= t[0]
    g = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "t": t, "freq": g("freq_hz"), "strength": g("strength_db"),
        "smoothed": g("smoothed_db"), "peak": g("peakhold_db"),
        "cadence": g("cadence_hz"), "conf": g("cadence_conf"),
        "locked": g("locked").astype(bool),
    }


def sparkline(vals, width=70):
    blocks = "▁▂▃▄▅▆▇█"
    v = np.array(vals, dtype=float)
    # resample to `width` columns by averaging
    idx = np.linspace(0, len(v), width + 1).astype(int)
    cols = [v[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else v[idx[i]]
            for i in range(width)]
    cols = np.array(cols)
    lo, hi = np.nanmin(cols), np.nanmax(cols)
    if hi - lo < 1e-9:
        hi = lo + 1
    norm = (cols - lo) / (hi - lo)
    return "".join(blocks[min(7, int(x * 7.999))] for x in norm), lo, hi


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("beephunter_log_*.csv"))[-1]
    d = load(path)
    t, dur = d["t"], d["t"][-1]
    n = len(t)
    print(f"=== {path} ===")
    print(f"samples: {n}   duration: {dur:.1f} s   rate: {n/max(dur,1):.1f} Hz\n")

    # --- lock / frequency / cadence health ---
    lk = d["locked"]
    print(f"locked:   {lk.mean()*100:4.1f}% of the time")
    fl = d["freq"][d["freq"] > 0]
    print(f"frequency: median {np.median(fl):.0f} Hz  "
          f"(IQR {np.percentile(fl,25):.0f}-{np.percentile(fl,75):.0f}, "
          f"min {fl.min():.0f}, max {fl.max():.0f})")
    cl = d["cadence"][d["conf"] > 0.3]
    if cl.size:
        print(f"cadence:   median {np.median(cl):.2f}/s  "
              f"({np.median(cl)*60:.0f} bpm) over confident frames")
    print()

    # --- strength timeline ---
    s = d["smoothed"]
    spark, lo, hi = sparkline(s)
    print("SMOOTHED STRENGTH over the walk (left=start, right=end):")
    print(f"  {hi:6.1f} dB ┐")
    print(f"  {spark}")
    print(f"  {lo:6.1f} dB ┘   each column ~{dur/70:.1f} s\n")

    # --- where was it loudest? top windows ---
    # smooth a touch more to find robust hotspots, then report top moments
    k = max(1, int(0.5 * n / max(dur, 1)))  # ~0.5 s window
    kernel = np.ones(k) / k
    ss = np.convolve(s, kernel, mode="same")
    order = np.argsort(ss)[::-1]
    print("STRONGEST MOMENTS (by smoothed band level):")
    picked = []
    for i in order:
        if all(abs(t[i] - t[j]) > 4.0 for j in picked):  # spread out hits
            picked.append(i)
            print(f"  t = {t[i]:6.1f} s   {s[i]:6.1f} dB   "
                  f"freq {d['freq'][i]:.0f} Hz  cad {d['cadence'][i]:.1f}/s  "
                  f"conf {d['conf'][i]:.2f}  {'LOCKED' if lk[i] else ''}")
        if len(picked) >= 5:
            break
    print()

    gi = int(np.argmax(ss))
    print(f"PEAK: strongest at t = {t[gi]:.1f} s  ->  {s[gi]:.1f} dB")
    print(f"weakest smoothed level: {s.min():.1f} dB   "
          f"dynamic range across walk: {s.max()-s.min():.1f} dB")
    print(f"\nInterpretation: the source is near wherever you were standing around "
          f"t≈{t[gi]:.0f}s. A large dynamic range ({s.max()-s.min():.0f} dB) means the walk "
          f"clearly passed closer/farther; a small range means you stayed roughly equidistant.")


if __name__ == "__main__":
    main()
