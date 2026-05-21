"""Render a static OpenStreetMap PNG with our GPS data overlaid.

Stitches OSM tiles (Pillow) for the bounding box and plots: the first walk
loop (4 waypoints), the stationary-test GPS cluster, and the home address.
No browser needed. Respect OSM tile policy: small volume, real User-Agent.
"""
import io, json, glob, math, os, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bh_config import HOME, CONTACT, WALK
DL = os.path.expanduser(r"~/Downloads")
UA = {"User-Agent": f"beephunter-dev/1.0 ({CONTACT})"}
Z = 19

def deg2num(lat, lon, z):
    n = 2**z; lr = math.radians(lat)
    x = (lon+180.0)/360.0*n
    y = (1.0 - math.asinh(math.tan(lr))/math.pi)/2.0*n
    return x, y

def fetch_tile(z, x, y):
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    req = urllib.request.Request(url, headers=UA)
    return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=20).read())).convert("RGB")

def stationary_points():
    pts = []
    for f in glob.glob(os.path.join(DL, "beephunter_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
            plat = "phone" if "Android" in d["meta"].get("ua","") else "laptop"
            for r in d["rows"]:
                if r.get("lat") is not None:
                    pts.append((r["lat"], r["lon"], plat))
        except Exception:
            pass
    return pts

def main():
    allpts = [HOME] + WALK + [(p[0],p[1]) for p in stationary_points()]
    lats = [p[0] for p in allpts]; lons = [p[1] for p in allpts]
    mlat = 0.00025; mlon = 0.00040   # margin
    latmin, latmax = min(lats)-mlat, max(lats)+mlat
    lonmin, lonmax = min(lons)-mlon, max(lons)+mlon

    x0f, y0f = deg2num(latmax, lonmin, Z)   # top-left
    x1f, y1f = deg2num(latmin, lonmax, Z)   # bottom-right
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))
    W = (x1-x0+1)*256; H = (y1-y0+1)*256
    canvas = Image.new("RGB", (W, H))
    for tx in range(x0, x1+1):
        for ty in range(y0, y1+1):
            try: canvas.paste(fetch_tile(Z, tx, ty), ((tx-x0)*256, (ty-y0)*256))
            except Exception as e: print("tile fail", tx, ty, e)

    def px(lat, lon):
        xf, yf = deg2num(lat, lon, Z)
        return (xf-x0)*256, (yf-y0)*256

    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=130)
    ax.imshow(np.asarray(canvas), origin="upper")

    # stationary cluster
    sp = stationary_points()
    for lat, lon, plat in sp:
        x, y = px(lat, lon)
        ax.plot(x, y, "o", ms=3, color=("#1f77b4" if plat=="phone" else "#d62728"), alpha=.5)
    # first walk loop
    wl = WALK + [WALK[0]]
    xs = [px(a,b)[0] for a,b in wl]; ys = [px(a,b)[1] for a,b in wl]
    ax.plot(xs, ys, "-", color="#ffb000", lw=2.5, label="first walk loop")
    for i,(a,b) in enumerate(WALK):
        x,y = px(a,b); ax.plot(x,y,"o",ms=7,color="#ffb000")
        ax.annotate(f"P{i}", (x,y), color="#7a5200", fontsize=11, fontweight="bold", xytext=(5,5), textcoords="offset points")
    # home
    hx, hy = px(*HOME)
    ax.plot(hx, hy, "*", ms=26, color="#19d219", markeredgecolor="k", markeredgewidth=1.2, label="HOME")
    ax.annotate("HOME #24", (hx,hy), color="#0a6b0a", fontsize=12, fontweight="bold", xytext=(8,-14), textcoords="offset points")

    # legend proxies
    ax.plot([],[], "o", color="#1f77b4", label="stationary test (phone)")
    ax.plot([],[], "o", color="#d62728", label="stationary test (laptop)")
    ax.legend(loc="lower right", fontsize=9, framealpha=.9)
    ax.set_title("BeepHunter: walk loop + stationary test", fontsize=12, fontweight="bold")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "street_map.png")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)

if __name__ == "__main__":
    main()
