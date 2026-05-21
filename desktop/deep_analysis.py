"""Deep analysis of a BeepHunter walk (hunter phone + stationary reference).

Pipeline:
  1. time-sync phone & reference (refine lag by cross-correlating frequency).
  2. remove the source's intrinsic frequency drift using the reference
     -> pure Doppler = phone_f - ref_f(t).
  3. reject multipath outliers (median filter + physical |v_radial|<=speed gate).
  4. find abeam zero-crossings on straight segments -> perpendicular lines
     -> least-squares source estimate.
  5. render an OSM map (track colored approaching/receding + abeam lines +
     source star + home) and a multi-panel diagnostic figure.
"""
import io, json, glob, math, os, sys, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bh_config import HOME, CONTACT
DL = os.path.expanduser(r"~/Downloads")
UA = {"User-Agent": f"beephunter-dev/1.0 ({CONTACT})"}
C = 343.0; Z = 19
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def newest(role):
    fs = sorted(glob.glob(os.path.join(DL, f"beephunter_{role}_*.json")), key=os.path.getmtime, reverse=True)
    return fs[0]

def load(f):
    d = json.load(open(f, encoding="utf-8")); r = d["rows"]
    g = lambda k: np.array([x.get(k) for x in r], dtype=float)
    return {"t": g("tw")/1000.0, "f": g("f"), "fraw": g("fraw"), "lat": g("lat"),
            "lon": g("lon"), "spd": g("spd"), "hdg": g("hdg"), "meta": d["meta"]}

def smooth(x, k):
    if k < 2: return x.copy()
    out = np.copy(x)
    for i in range(len(x)):
        lo, hi = max(0, i-k), min(len(x), i+k+1)
        seg = x[lo:hi]; seg = seg[np.isfinite(seg)]
        out[i] = np.mean(seg) if len(seg) else np.nan
    return out

def medfilt(x, k):
    out = np.copy(x)
    for i in range(len(x)):
        lo, hi = max(0, i-k), min(len(x), i+k+1)
        seg = x[lo:hi]; seg = seg[np.isfinite(seg)]
        out[i] = np.median(seg) if len(seg) else np.nan
    return out

# ---- OSM tiles ----
def deg2num(lat, lon, z):
    n = 2**z; return (lon+180)/360*n, (1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n
def fetch_tile(z, x, y):
    req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers=UA)
    return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=20).read())).convert("RGB")

def main():
    P = load(newest("hunter")); R = load(newest("reference"))
    # common grid over phone span
    dt = 0.2
    t0 = max(P["t"][0], R["t"][0]); t1 = min(P["t"][-1], R["t"][-1])
    g = np.arange(0, t1-t0, dt)
    pf = np.interp(g, P["t"]-t0, P["f"]); rf = np.interp(g, R["t"]-t0, R["f"])
    spd = np.interp(g, P["t"]-t0, np.nan_to_num(P["spd"], nan=0.0))
    lat = np.interp(g, P["t"]-t0, P["lat"]); lon = np.interp(g, P["t"]-t0, P["lon"])
    hdg = np.interp(g, P["t"]-t0, np.nan_to_num(P["hdg"], nan=0.0))

    # Clocks are NTP-synced (devices started ~1 s apart). The phone frequency is
    # dominated by Doppler, NOT the drift, so we must NOT sync on frequency.
    # Smooth the reference to keep only the slow intrinsic drift (removing its
    # ~1.5 Hz measurement noise) before subtracting it from the phone.
    rf_slow = smooth(rf, int(8/dt))
    f0 = np.nanmedian(rf_slow)

    # --- Doppler: differential (drift-removed) vs phone-only ---
    dop_diff = pf - rf_slow
    rest_solo = np.nanmedian(pf[spd<0.5]) if np.any(spd<0.5) else np.nanmedian(pf)
    dop_solo = pf - rest_solo

    # outlier rejection: physical gate |v_radial| <= 1.6*speed, then median filter
    vmax = 1.6*np.maximum(spd, 0.3)*f0/C + 2.0  # allow margin + 2 Hz noise
    dop = np.where(np.abs(dop_diff) <= vmax, dop_diff, np.nan)
    dop = medfilt(dop, 3)
    dop = smooth(dop, int(1.2/dt))
    v_r = dop*C/f0

    drift = np.nanmax(rf_slow)-np.nanmin(rf_slow)
    print(f"overlap {t1-t0:.0f}s | wall-clock sync (no freq roll)")
    print(f"rest f0={f0:.2f} Hz | reference intrinsic drift during walk = {drift:.1f} Hz")
    print(f"phone speed median {np.nanmedian(spd):.2f} max {np.nanmax(spd):.2f} m/s")
    print(f"Doppler(diff) swing {np.nanmin(dop_diff):.1f}..{np.nanmax(dop_diff):.1f} Hz; cleaned {np.nanmin(dop):.1f}..{np.nanmax(dop):.1f}")

    # ---- abeam crossings on STRAIGHT, moving, strongly-swinging segments ----
    P0lat, P0lon = lat.mean(), lon.mean()
    mlat, mlon = 111320.0, 111320.0*math.cos(math.radians(P0lat))
    xy = lambda la, lo: ((lo-P0lon)*mlon, (la-P0lat)*mlat)
    ll = lambda x, y: (P0lat+y/mlat, P0lon+x/mlon)
    X = (lon-P0lon)*mlon; Y = (lat-P0lat)*mlat
    # velocity unit vector from the SMOOTHED track (GPS heading is too noisy)
    Xs, Ys = smooth(X, int(1.0/dt)), smooth(Y, int(1.0/dt))
    vx, vy = np.gradient(Xs, dt), np.gradient(Ys, dt)
    sp_tr = np.hypot(vx, vy)
    lines=[]; cross=[]; last_i=-9999
    W=int(4/dt)
    for i in range(3, len(dop)-3):
        if not (np.isfinite(dop[i-1]) and np.isfinite(dop[i])): continue
        if (dop[i-1]>0) != (dop[i]>0):
            seg = dop[max(0,i-W):min(len(dop),i+W)]; seg=seg[np.isfinite(seg)]
            # require a STRONG clean swing (real pass), not noise near zero
            if len(seg)<8 or np.nanmax(seg)<4 or np.nanmin(seg)>-4: continue
            if sp_tr[i]<0.6: continue
            if i-last_i < int(5/dt): continue          # de-dup over 5 s
            last_i=i
            n=(vx[i]/sp_tr[i], vy[i]/sp_tr[i]); p=(X[i],Y[i])
            cross.append((i,p,n)); lines.append((p,n))

    # LS intersection of abeam lines
    src=None; cond=None
    if len(lines)>=2:
        A=np.zeros((2,2)); bb=np.zeros(2)
        for p,nrm in lines:
            nx,ny=nrm; c=nx*p[0]+ny*p[1]
            A+=np.array([[nx*nx,nx*ny],[ny*nx,ny*ny]]); bb+=np.array([nx*c,ny*c])
        cond=np.linalg.cond(A)
        try:
            q=np.linalg.solve(A,bb); src=ll(q[0],q[1])
        except Exception: pass
    print(f"abeam crossings: {len(lines)} | LS condition number: {cond}")
    if src:
        dh=math.hypot((src[0]-HOME[0])*mlat,(src[1]-HOME[1])*mlon)
        print(f"SOURCE ESTIMATE: {src[0]:.6f}, {src[1]:.6f}  ({dh:.0f} m from home)")
    for j,(i,p,n) in enumerate(cross):
        print(f"  abeam {j+1}: t={g[i]:.0f}s pos=({p[0]:+.0f},{p[1]:+.0f})m heading_vec=({n[0]:+.2f},{n[1]:+.2f})")

    # =================== FIGURE 1: diagnostics ===================
    fig, ax = plt.subplots(2,2, figsize=(14,9)); fig.suptitle("BeepHunter walk — Doppler diagnostics", fontweight="bold")
    a1=ax[0,0]; a1.plot(g,pf,lw=.7,color="#1f77b4",label="phone f"); a1.plot(g,rf_slow,lw=1.6,color="#d62728",label="reference f (drift, smoothed)")
    a1.set_title(f"Frequency vs time (source drifts {drift:.1f} Hz during walk)"); a1.set_ylabel("Hz"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2=ax[0,1]; a2.plot(g,dop_diff,lw=.5,color="#bbb",label="raw differential")
    a2.plot(g,dop,lw=1.6,color="#2ca02c",label="cleaned Doppler"); a2.axhline(0,color="k",lw=.6)
    a2.plot(g,dop_solo,lw=.6,color="#ff7f0e",alpha=.6,label="phone-only (drift NOT removed)")
    a2.set_title("Doppler shift — reference removes the drift"); a2.set_ylabel("Hz"); a2.legend(fontsize=8); a2.grid(alpha=.3)
    a3=ax[1,0]; a3.plot(g,spd,color="#555",label="GPS speed"); a3.plot(g,np.abs(v_r),color="#2ca02c",label="|radial v| from Doppler")
    a3.set_title("Sanity: |radial velocity| must stay under GPS speed"); a3.set_ylabel("m/s"); a3.set_xlabel("s"); a3.legend(fontsize=8); a3.grid(alpha=.3)
    a4=ax[1,1]; sc=a4.scatter(X,Y,c=np.clip(dop,-8,8),cmap="coolwarm",s=8)
    for i,p,n in cross: a4.plot(p[0],p[1],"o",ms=9,mfc="yellow",mec="k")
    hx,hy=xy(*HOME); a4.plot(hx,hy,"*",ms=20,color="#19d219",mec="k",label="home")
    if src: sx,sy=xy(*src); a4.plot(sx,sy,"X",ms=16,color="lime",mec="k",label="source est")
    a4.set_aspect("equal"); a4.set_title("Track (m), colored by Doppler; ● abeam"); a4.legend(fontsize=8); a4.grid(alpha=.3); plt.colorbar(sc,ax=a4,label="Hz")
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(os.path.join(ROOT,"walk_diagnostics.png"),dpi=120)
    print("saved walk_diagnostics.png")

    # =================== FIGURE 2: OSM map ===================
    lats=list(lat)+[HOME[0]]; lons=list(lon)+[HOME[1]]
    la0,la1=min(lats)-.0002,max(lats)+.0002; lo0,lo1=min(lons)-.0003,max(lons)+.0003
    x0f,y0f=deg2num(la1,lo0,Z); x1f,y1f=deg2num(la0,lo1,Z)
    x0,y0,x1,y1=int(x0f),int(y0f),int(x1f),int(y1f)
    W,H=(x1-x0+1)*256,(y1-y0+1)*256; canvas=Image.new("RGB",(W,H))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try: canvas.paste(fetch_tile(Z,tx,ty),((tx-x0)*256,(ty-y0)*256))
            except Exception as e: print("tile",tx,ty,e)
    px=lambda la,lo:((deg2num(la,lo,Z)[0]-x0)*256,(deg2num(la,lo,Z)[1]-y0)*256)
    fig2,axm=plt.subplots(figsize=(W/110,H/110),dpi=130); axm.imshow(np.asarray(canvas),origin="upper")
    pxs=[px(lat[i],lon[i]) for i in range(len(g))]
    cvals=np.clip(np.nan_to_num(dop),-8,8)
    for i in range(1,len(pxs)):
        col=plt.cm.coolwarm((cvals[i]+8)/16)
        axm.plot([pxs[i-1][0],pxs[i][0]],[pxs[i-1][1],pxs[i][1]],"-",color=col,lw=4)
    for i,p,n in cross:
        cx,cy=px(lat[i],lon[i]); axm.plot(cx,cy,"o",ms=10,mfc="yellow",mec="k")
        perp=(-n[1],n[0])
        e1=ll(p[0]+perp[0]*45,p[1]+perp[1]*45); e2=ll(p[0]-perp[0]*45,p[1]-perp[1]*45)
        p1=px(*e1); p2=px(*e2); axm.plot([p1[0],p2[0]],[p1[1],p2[1]],"--",color="gold",lw=1.5)
    hx,hy=px(*HOME); axm.plot(hx,hy,"*",ms=24,color="#19d219",mec="k",mew=1.2)
    axm.annotate("HOME #24",(hx,hy),color="#0a6b0a",fontsize=11,fontweight="bold",xytext=(8,-12),textcoords="offset points")
    if src:
        sx,sy=px(*src); axm.plot(sx,sy,"X",ms=20,color="lime",mec="k",mew=2)
        axm.annotate("source est",(sx,sy),color="#0a6b0a",fontsize=11,fontweight="bold",xytext=(8,8),textcoords="offset points")
    axm.set_title("BeepHunter walk — Doppler abeam lines on OSM (red=approaching, blue=receding)",fontsize=11,fontweight="bold")
    axm.set_xlim(0,W); axm.set_ylim(H,0); axm.axis("off")
    fig2.tight_layout(); fig2.savefig(os.path.join(ROOT,"walk_map.png"),dpi=130,bbox_inches="tight")
    print("saved walk_map.png")

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
