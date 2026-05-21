"""Map a BeepHunter walk onto GPS waypoints and estimate the source location.

Assumes uniform walking speed around the closed loop of waypoints, so each log
sample's timestamp maps to a fractional distance along the path. Colors the
path by signal strength, fits a 1/r free-field source model by least squares,
and writes an interactive Leaflet map (map.html).
"""
import csv
import glob
import json
import sys
import webbrowser
from datetime import datetime
from math import cos, radians

import numpy as np
from scipy.optimize import least_squares

# --- GPS waypoints of the walked loop (lat, lon), closed back to start ------
WAYPOINTS = [
    (51.028244, 3.727147),  # P0  (start/end)
    (51.028157, 3.726526),  # P1
    (51.028093, 3.726618),  # P2
    (51.028177, 3.727106),  # P3
]


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    t = np.array([datetime.fromisoformat(r["iso_time"]).timestamp() for r in rows])
    t -= t[0]
    g = lambda k: np.array([float(r[k]) for r in rows])
    return t, g("freq_hz"), g("strength_db"), g("smoothed_db"), g("cadence_hz"), g("conf" if "conf" in rows[0] else "cadence_conf")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("beephunter_log_*.csv"))[-1]
    t, freq, strength, smoothed, cadence, conf = load(path)
    dur = t[-1]

    # --- local equirectangular projection (meters) around the centroid ------
    lat0 = np.mean([p[0] for p in WAYPOINTS])
    lon0 = np.mean([p[1] for p in WAYPOINTS])
    mlat = 111320.0
    mlon = 111320.0 * cos(radians(lat0))
    to_xy = lambda la, lo: ((lo - lon0) * mlon, (la - lat0) * mlat)
    to_ll = lambda x, y: (lat0 + y / mlat, lon0 + x / mlon)

    loop = WAYPOINTS + [WAYPOINTS[0]]
    pts = np.array([to_xy(la, lo) for la, lo in loop])  # (5,2)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    perim = cum[-1]
    speed = perim / dur
    print(f"=== {path} ===")
    print(f"loop perimeter {perim:.1f} m, duration {dur:.1f} s -> "
          f"assumed speed {speed:.2f} m/s")
    for i in range(4):
        print(f"  P{i}->P{(i+1)%4}: {seglen[i]:5.1f} m")

    # --- map each sample time -> position along the loop --------------------
    def pos_at(frac):
        d = np.clip(frac, 0, 1) * perim
        seg = np.searchsorted(cum, d, side="right") - 1
        seg = np.clip(seg, 0, 3)
        local = (d - cum[seg]) / seglen[seg]
        p = pts[seg] + local[:, None] * (pts[seg + 1] - pts[seg])
        return p[:, 0], p[:, 1]

    X, Y = pos_at(t / dur)

    # --- least-squares source localization: dB = L0 - 20*log10(r+1) ---------
    good = (strength > -100) & (freq > 0)
    w = np.sqrt(conf[good] + 0.05)
    Xg, Yg, Sg = X[good], Y[good], strength[good]

    # init: power-weighted centroid
    p_lin = 10 ** (Sg / 10.0)
    sx0 = np.sum(Xg * p_lin) / np.sum(p_lin)
    sy0 = np.sum(Yg * p_lin) / np.sum(p_lin)

    def resid(params):
        sx, sy, L0 = params
        r = np.sqrt((Xg - sx) ** 2 + (Yg - sy) ** 2)
        return (L0 - 20 * np.log10(r + 1.0) - Sg) * w

    res = least_squares(resid, [sx0, sy0, Sg.max() + 10], method="trf")
    sx, sy, L0 = res.x
    src_lat, src_lon = to_ll(sx, sy)

    # distance from estimate to each waypoint
    print(f"\nPower-weighted centroid of walk : {to_ll(sx0, sy0)}")
    print(f"Least-squares SOURCE estimate    : {src_lat:.6f}, {src_lon:.6f}")
    print(f"  fit residual RMS: {np.sqrt(np.mean(res.fun**2)):.1f} dB")
    for i, (la, lo) in enumerate(WAYPOINTS):
        d = np.hypot(*(np.array(to_xy(la, lo)) - [sx, sy]))
        print(f"  {d:6.1f} m from P{i} ({la}, {lo})")

    # --- write Leaflet map --------------------------------------------------
    # downsample samples for clarity
    step = max(1, len(t) // 250)
    samp = [{"lat": to_ll(X[i], Y[i])[0], "lon": to_ll(X[i], Y[i])[1],
             "s": round(float(strength[i]), 1), "t": round(float(t[i]), 1)}
            for i in range(0, len(t), step)]
    data = {
        "waypoints": [{"lat": p[0], "lon": p[1]} for p in WAYPOINTS],
        "samples": samp,
        "source": {"lat": src_lat, "lon": src_lon},
        "smin": float(np.percentile(strength[good], 5)),
        "smax": float(np.percentile(strength[good], 99)),
        "center": [lat0, lon0],
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    open("map.html", "w", encoding="utf-8").write(html)
    print("\nWrote map.html")


HTML_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>BeepHunter map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0} .lg{background:#fff;padding:8px 10px;
border-radius:6px;font:13px sans-serif;line-height:1.5;box-shadow:0 1px 5px rgba(0,0,0,.4)}</style>
</head><body><div id="map"></div><script>
const D = __DATA__;
const map = L.map('map',{maxZoom:21}).setView(D.center, 19);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxNativeZoom:19, maxZoom:21, keepBuffer:8, attribution:'© OpenStreetMap'}).addTo(map);
setTimeout(()=>map.invalidateSize(), 300);

function color(s){ // blue (weak) -> red (strong)
  let f = (s - D.smin)/(D.smax - D.smin); f = Math.max(0,Math.min(1,f));
  const r = Math.round(255*f), b = Math.round(255*(1-f));
  return `rgb(${r},${Math.round(80*(1-Math.abs(f-0.5)*2))},${b})`;
}
// walked path colored by strength
for(let i=1;i<D.samples.length;i++){
  const a=D.samples[i-1], b=D.samples[i];
  L.polyline([[a.lat,a.lon],[b.lat,b.lon]],{color:color(b.s),weight:7,opacity:.9}).addTo(map);
}
// strength dots
D.samples.forEach(p=>{
  L.circleMarker([p.lat,p.lon],{radius:4,color:'#222',weight:.5,fillColor:color(p.s),
    fillOpacity:.95}).addTo(map).bindTooltip(`${p.s} dB @ ${p.t}s`);
});
// waypoints
D.waypoints.forEach((p,i)=>{
  L.marker([p.lat,p.lon]).addTo(map).bindTooltip('P'+i,{permanent:true});
});
// estimated source
L.circle([D.source.lat,D.source.lon],{radius:4,color:'#000',fillColor:'#ff0',fillOpacity:1})
  .addTo(map).bindTooltip('★ estimated source',{permanent:true});
L.marker([D.source.lat,D.source.lon]).addTo(map);

const lg = L.control({position:'topright'});
lg.onAdd = ()=>{ const d=L.DomUtil.create('div','lg');
  d.innerHTML = `<b>BeepHunter</b><br>🔵 weak → 🔴 strong<br>★ estimated source<br>`+
    `${D.source.lat.toFixed(6)}, ${D.source.lon.toFixed(6)}`; return d; };
lg.addTo(map);
</script></body></html>"""


if __name__ == "__main__":
    main()
