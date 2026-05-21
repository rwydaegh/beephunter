"""BeepHunter master overview: one self-contained interactive HTML with many
toggleable data layers (Doppler, confidence, band level, input level, cadence,
frequency, speed), per-sample click popups, walk-direction arrows, abeam
lines-of-position, and a point-source Doppler fit. Map view is fit EXACTLY to
the data extent (zoomSnap:0 + zero padding) so there is no tile slack.

Also prints a data-quality / cleanliness report to the console.
"""
import glob, json, math, os, sys
import numpy as np
from scipy.optimize import least_squares

from bh_config import HOME, HOMELABEL
DL = os.path.expanduser(r"~/Downloads"); C = 343.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def newest(role): return sorted(glob.glob(os.path.join(DL, f"beephunter_{role}_*.json")), key=os.path.getmtime)[-1]
def load(f):
    d = json.load(open(f, encoding="utf-8")); r = d["rows"]; g = lambda k: np.array([x.get(k) for x in r], float)
    return {k: g(k) for k in ["tw","f","fraw","conf","lvl","inlvl","cad","ccf","lat","lon","acc","spd","hdg"]}
def sm(x, k):
    o = np.copy(x)
    for i in range(len(x)):
        s = x[max(0,i-k):min(len(x),i+k+1)]; s = s[np.isfinite(s)]; o[i] = np.mean(s) if len(s) else np.nan
    return o
def mf(x, k):
    o = np.copy(x)
    for i in range(len(x)):
        s = x[max(0,i-k):min(len(x),i+k+1)]; s = s[np.isfinite(s)]; o[i] = np.median(s) if len(s) else np.nan
    return o

def twrange(f):
    tw=np.array([x.get("tw") for x in json.load(open(f,encoding="utf-8"))["rows"]],float)/1000
    tw=tw[np.isfinite(tw)]; return (tw[0],tw[-1]) if len(tw) else (0.0,0.0)

def wid(h): return os.path.basename(h).replace("beephunter_hunter_hunter_","").replace(".json","")

def usable_walks(hunters, refs):
    """Walks with a real tone lock AND an overlapping reference -> selectable measurements."""
    out = []
    for h in hunters:
        a, b = twrange(h)
        fok = np.isfinite(np.array([x.get("f") for x in json.load(open(h,encoding="utf-8"))["rows"]], float)).mean()*100
        if fok < 5: continue                              # no tone lock -> no Doppler possible
        Rf = next((r for r in refs if twrange(r)[0] < b and twrange(r)[1] > a), None)
        if Rf is None: continue                           # no simultaneous reference -> can't subtract drift
        wi = wid(h)
        out.append({"id": wi, "label": f"{wi[11:]} · {b-a:.0f}s · tone {fok:.0f}%", "Pf": h, "Rf": Rf})
    return out

def analyze(Pf, Rf):
    """Compute every overlay layer for one hunter/reference pair; return the JS data dict."""
    P = load(Pf); R = load(Rf)
    print(f"using hunter = {os.path.basename(Pf)}\n      reference = {os.path.basename(Rf)}")
    dt = 0.2
    t0 = max(P["t" if False else "tw"][0], R["tw"][0]) / 1000  # placeholder
    pt = P["tw"]/1000; rt = R["tw"]/1000
    t0 = max(pt[0], rt[0]); t1 = min(pt[-1], rt[-1]); g = np.arange(0, t1 - t0, dt)
    I = lambda src, key: np.interp(g, src["tw"]/1000 - t0, src[key])
    pf = I(P,"f")
    # smooth the reference in its OWN full timeline (incl pre/post-roll margin) so the
    # 8 s kernel stays two-sided at the walk edges, THEN sample onto the walk grid.
    gr = np.arange(0, rt[-1]-rt[0], dt)
    rf_full = sm(np.interp(gr, rt-rt[0], R["f"]), int(8/dt))
    rf = np.interp(g + (t0-rt[0]), gr, rf_full); f0 = np.nanmedian(rf)
    conf = I(P,"conf"); lvl = I(P,"lvl"); inlvl = I(P,"inlvl"); cad = I(P,"cad"); ccf = I(P,"ccf")
    spd = np.interp(g, pt - t0, np.nan_to_num(P["spd"], nan=0))
    lat = I(P,"lat"); lon = I(P,"lon"); acc = I(P,"acc")
    latS = sm(lat, int(1.5/dt)); lonS = sm(lon, int(1.5/dt))  # de-jittered display track
    mlat = 111320.0; mlon = 111320.0 * math.cos(math.radians(np.nanmean(lat)))
    X = (lon - lon.mean()) * mlon; Y = (lat - lat.mean()) * mlat
    Xs, Ys = sm(X, int(1/dt)), sm(Y, int(1/dt)); vx, vy = np.gradient(Xs, dt), np.gradient(Ys, dt); sp = np.hypot(vx, vy)

    dop_raw = pf - rf
    vmax = 1.6 * np.maximum(spd, .3) * f0 / C + 2.0
    dop = np.where(np.abs(dop_raw) <= vmax, dop_raw, np.nan); dop = sm(mf(dop, 3), int(1.2/dt))

    # ---------- cleanliness / quality report ----------
    print("="*64); print("DATA QUALITY / CLEANLINESS"); print("="*64)
    print(f"hunter   : {os.path.basename(Pf)}  ({(pt[-1]-pt[0]):.0f}s)")
    print(f"reference: {os.path.basename(Rf)}  ({(rt[-1]-rt[0]):.0f}s)")
    print(f"overlap window analysed: {g[-1]:.0f}s")
    print(f"reference confidence median {np.nanmedian(R['conf']):.2f}  (clean tone baseline)")
    print(f"reference drift over session: {np.nanmedian(R['f'][:60]):.1f} -> {np.nanmedian(R['f'][-60:]):.1f} Hz "
          f"({(np.nanmedian(R['f'][-60:])-np.nanmedian(R['f'][:60]))/f0*1e6:+.0f} ppm)")
    tc = conf > 0.30
    print(f"hunter trustworthy (conf>0.30): {tc.mean()*100:.0f}%  | conf<0.15 (junk): {(conf<0.15).mean()*100:.0f}%")
    fm = np.isfinite(lvl) & np.isfinite(conf)
    print(f"corr(band level, conf) = {np.corrcoef(lvl[fm], conf[fm])[0,1]:+.2f}  "
          "(NEGATIVE => loud spots are dirtier, not closer)")
    # cross-device clock sanity: the Doppler subtraction TRUSTS wall-clock (NTP) alignment.
    # (start-time delta is dominated by intentional pre-roll, so it can't reveal sub-second skew.)
    print("clock sync: Doppler subtraction trusts wall-clock (NTP) alignment; sub-second skew "
          "is not measurable from these logs and is NOT auto-corrected.")
    rest = spd < 0.3
    if rest.sum() > 10:
        print(f"  end-to-end sanity: median |Doppler| while stationary (spd<0.3) = "
              f"{np.nanmedian(np.abs(dop[rest])):.2f} Hz  (~0 Hz => clocks+subtraction are clean)")
    qs = np.nanpercentile(lvl, [0,25,50,75,100])
    for i in range(4):
        m = (lvl>=qs[i])&(lvl<=qs[i+1])
        print(f"   level [{qs[i]:.0f},{qs[i+1]:.0f}]dB: median conf {np.nanmedian(conf[m]):.2f}, trustworthy {(conf[m]>0.3).mean()*100:.0f}%")

    # ---------- point-source Doppler fit on trustworthy moving samples ----------
    m = np.isfinite(dop) & (sp > 0.6) & (conf > 0.30)
    Xm, Ym, vxm, vym, dm = X[m], Y[m], vx[m], vy[m], dop[m]
    def resid(S):
        dx = S[0]-Xm; dy = S[1]-Ym; r = np.hypot(dx,dy)+1e-6
        return (f0/C)*(vxm*dx+vym*dy)/r - dm
    best = None
    for gx in range(-80,81,20):
        for gy in range(-80,81,20):
            try:
                rr = least_squares(resid,[gx,gy],method="lm",max_nfev=2000)
                if best is None or rr.cost < best.cost: best = rr
            except Exception: pass
    S = best.x; res = resid(S); rms = math.sqrt(np.mean(res**2))
    R2 = 1 - np.sum(res**2)/np.sum((dm-dm.mean())**2)
    slat = lat.mean()+S[1]/mlat; slon = lon.mean()+S[0]/mlon
    dh = math.hypot((slat-HOME[0])*mlat,(slon-HOME[1])*mlon)
    print("-"*64)
    print(f"point-source fit (trustworthy moving samples, n={m.sum()}):")
    print(f"  S=({S[0]:+.0f},{S[1]:+.0f})m -> {slat:.6f},{slon:.6f}  | {dh:.0f} m from home")
    print(f"  RMS resid {rms:.2f} Hz vs Doppler std {dm.std():.2f} Hz | R2={R2:.2f}")
    verdict = "TRUSTWORTHY" if R2>0.5 else ("WEAK" if R2>0.25 else "NOT SUPPORTED (multipath)")
    print(f"  verdict: {verdict}")

    # ---------- abeam lines-of-position (source is perpendicular to velocity here) ----------
    lops = []
    sgn = np.sign(dop); cross = np.where(np.isfinite(dop[:-1]) & np.isfinite(dop[1:]) & (sgn[:-1] != sgn[1:]))[0]
    last = -1e9
    for i in cross:
        w = slice(max(0,i-int(4/dt)), min(len(g),i+int(4/dt)))
        seg = dop[w]; seg = seg[np.isfinite(seg)]
        if len(seg)<5 or np.nanmax(seg) < 3 or np.nanmin(seg) > -3: continue   # need real swing
        if sp[i] < 0.6 or conf[i] < 0.3: continue
        if g[i]-last < 4: continue
        last = g[i]
        # perpendicular-to-velocity line of position through this point
        ang = math.atan2(vx[i], vy[i])  # bearing of velocity
        perp = ang + math.pi/2
        L = 45.0
        dN = L*math.cos(perp); dE = L*math.sin(perp)
        a = (lat[i]-dN/mlat, lon[i]-dE/mlon); b = (lat[i]+dN/mlat, lon[i]+dE/mlon)
        lops.append({"a":[round(a[0],7),round(a[1],7)],"b":[round(b[0],7),round(b[1],7)],"t":round(g[i],0)})
    print(f"abeam lines-of-position (perpendicular swings): {len(lops)}")

    # ---------- local Doppler source-bearing field (vector per spatial cell) ----------
    # In each cell, solve v . u = (c/f0)*doppler for the unit source-direction u.
    # Needs >=2 non-parallel headings in the cell (else the cross-street component
    # is unresolvable -> we skip the cell). |u|~1 and high local R^2 => trustworthy bearing.
    xs, ys, vxs, vys, ds = X[m], Y[m], vx[m], vy[m], dop[m]
    cell = 6.0; field = []
    if len(xs) > 20:
        xe = np.arange(math.floor(xs.min()/cell)*cell, xs.max()+cell, cell)
        ye = np.arange(math.floor(ys.min()/cell)*cell, ys.max()+cell, cell)
        for xi in xe:
            for yi in ye:
                c = (xs>=xi)&(xs<xi+cell)&(ys>=yi)&(ys<yi+cell)
                if c.sum() < 6: continue
                A = np.column_stack([vxs[c], vys[c]]); sv = np.linalg.svd(A, compute_uv=False)
                if sv[0] < 1e-6 or sv[-1] < 0.30*sv[0]: continue          # only one heading -> can't resolve a vector
                b = ds[c]*C/f0
                u, *_ = np.linalg.lstsq(A, b, rcond=None); mag = math.hypot(*u)
                if mag < 1e-3: continue
                pred = A@u; sstot = np.sum((b-b.mean())**2)
                r2c = 1 - np.sum((b-pred)**2)/sstot if sstot > 1e-9 else 0.0
                cx, cy = xi+cell/2, yi+cell/2
                field.append({"lat":round(lat.mean()+cy/mlat,7),"lon":round(lon.mean()+cx/mlon,7),
                              "ux":round(u[0]/mag,3),"uy":round(u[1]/mag,3),
                              "mag":round(float(mag),2),"r2":round(float(r2c),2),"n":int(c.sum())})
    print(f"Doppler bearing field: {len(field)} resolvable cells, "
          f"{sum(1 for q in field if q['r2']>0.3)} confident (local R2>0.3)")

    # ---------- per-sample Doppler arrows (red=approaching/forward, blue=receding/back) ----------
    XsD=(lonS-lon.mean())*mlon; YsD=(latS-lat.mean())*mlat
    arc=np.concatenate([[0],np.cumsum(np.hypot(np.diff(XsD),np.diff(YsD)))])
    dmq=np.nanpercentile(np.abs(dop[np.isfinite(dop)]),95); Lq=11.0; dq=[]; nxt=0.0; ARC=0.6
    for i in range(len(g)):
        if arc[i]>=nxt and np.isfinite(dop[i]) and sp[i]>0.6 and conf[i]>0.30:
            hxu,hyu=vx[i]/sp[i],vy[i]/sp[i]; s=float(np.clip(dop[i]/dmq,-1.4,1.4))*Lq
            dq.append({"lat":round(latS[i],7),"lon":round(lonS[i],7),
                       "ela":round(latS[i]+(hyu*s)/mlat,7),"elo":round(lonS[i]+(hxu*s)/mlon,7),
                       "d":round(float(dop[i]),2)})
            nxt=arc[i]+ARC
    print(f"per-sample Doppler arrows: {len(dq)} (every {ARC} m; cursor-revealed; max arrow = {dmq:.1f} Hz)")

    # ---------- emit HTML ----------
    step = max(1, len(g)//650)
    def r2(v): return round(float(v),2) if np.isfinite(v) else None
    pts = []
    for i in range(0,len(g),step):
        pts.append({"lat":round(latS[i],7),"lon":round(lonS[i],7),
                    "rla":round(lat[i],7),"rlo":round(lon[i],7),"t":round(g[i],1),
                    "d":r2(dop[i]),"cf":r2(conf[i]),"lv":r2(lvl[i]),"il":r2(inlvl[i]),
                    "cd":r2(cad[i]),"cc":r2(ccf[i]),"fq":r2(pf[i]),"sp":r2(spd[i]),"ac":r2(acc[i])})
    fin = lambda a: a[np.isfinite(a)]
    dom = {
        "dopMax": round(float(np.nanpercentile(np.abs(fin(dop)),95)),1),
        "lvLo": round(float(np.nanpercentile(fin(lvl),5)),0), "lvHi": round(float(np.nanpercentile(fin(lvl),95)),0),
        "ilLo": round(float(np.nanpercentile(fin(inlvl),5)),0), "ilHi": round(float(np.nanpercentile(fin(inlvl),95)),0),
        "fqMed": round(float(f0),1), "fqHalf": round(float(np.nanpercentile(np.abs(fin(pf)-f0),95)),1),
        "spHi": round(float(np.nanpercentile(fin(spd),95)),2),
    }
    data = {"pts":pts,"home":list(HOME),"src":[round(slat,7),round(slon,7)],
            "R2":round(R2,2),"rms":round(rms,2),"f0":round(f0,1),"verdict":verdict,
            "dh":round(dh,0),"lops":lops,"dom":dom,"field":field,"dquiver":dq,"mlat":mlat,"mlon":mlon,
            "bbox":[[float(np.nanmin(latS)),float(np.nanmin(lonS))],[float(np.nanmax(latS)),float(np.nanmax(lonS))]]}
    return data

def main():
    import contextlib, io
    sel = sys.argv[1] if len(sys.argv) > 1 else None       # optional: substring of a walk id, e.g. "15-02"
    hunters = sorted(glob.glob(os.path.join(DL,"beephunter_hunter_*.json")),    key=os.path.getmtime, reverse=True)
    refs    = sorted(glob.glob(os.path.join(DL,"beephunter_reference_*.json")), key=os.path.getmtime, reverse=True)
    walks = usable_walks(hunters, refs)
    if not walks: sys.exit("no usable walks (need a tone lock + an overlapping reference recording)")
    print("usable measurements:\n  " + "\n  ".join(f"{w['label']}   [{w['id']}]" for w in walks))
    cur = next((w for w in walks if sel in w["id"]), walks[0]) if sel else walks[0]
    meta = [{"id": w["id"], "label": w["label"], "href": f"overview_{w['id']}.html"} for w in walks]
    # Build one self-contained page per walk; the in-page dropdown switches between them by navigation.
    for w in walks:
        if w["id"] == cur["id"]:
            print("="*64); print(f"SELECTED: {w['id']}"); data = analyze(w["Pf"], w["Rf"])
        else:
            with contextlib.redirect_stdout(io.StringIO()): data = analyze(w["Pf"], w["Rf"])  # quiet for the rest
        data["walks"] = meta; data["current"] = w["id"]; data["homeLabel"] = HOMELABEL
        html = TEMPLATE.replace("__DATA__", json.dumps(data))
        open(os.path.join(ROOT, f"overview_{w['id']}.html"), "w", encoding="utf-8").write(html)
        if w["id"] == cur["id"]:
            open(os.path.join(ROOT, "overview.html"), "w", encoding="utf-8").write(html)
    print("-"*64)
    print(f"saved overview.html (= {cur['id']}) + overview_<id>.html for {len(walks)} walks; switch via the in-page dropdown")

TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>BeepHunter overview</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body,#map{height:100%;margin:0}
.panel{font:12px system-ui,sans-serif;line-height:1.5}
.leaflet-control-layers{font:13px system-ui,sans-serif}
.lg{background:#fff;padding:8px 11px;border-radius:7px;box-shadow:0 1px 6px rgba(0,0,0,.4);max-width:230px}
.lg b{font-size:13px}
.bar{height:11px;border-radius:3px;margin:5px 0 2px;background:linear-gradient(to right,var(--g))}
.bl{display:flex;justify-content:space-between;font-size:10px;color:#444}
.arw div{font-size:15px;line-height:15px;color:#111;text-shadow:0 0 2px #fff}
.note{font-size:11px;color:#555;margin-top:6px;border-top:1px solid #ddd;padding-top:5px}
</style></head><body><div id="map"></div><script>
const D=__DATA__, P=D.pts, M=D.dom;
const map=L.map('map',{zoomSnap:0,maxZoom:28,preferCanvas:true});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxNativeZoom:19,maxZoom:28,attribution:'© OpenStreetMap (over-zoomed past z19)'}).addTo(map);

// ---- colour helpers ----
function lerp(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function ramp(stops,t){t=Math.max(0,Math.min(1,t));const x=t*(stops.length-1),i=Math.floor(x);
 if(i>=stops.length-1)return stops[stops.length-1];const c=lerp(stops[i],stops[i+1],x-i);return `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;}
const VIR=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
const INF=[[10,5,30],[87,16,110],[188,55,84],[249,142,9],[252,255,164]];
const RYG=[[200,40,40],[240,200,70],[40,160,70]];          // red->yellow->green (confidence)
function div(v,m){const t=Math.max(-1,Math.min(1,v/m));     // blue(-)..white(0)..red(+)
 if(t<0)return ramp([[40,90,235],[235,235,235]],t+1); return ramp([[235,235,235],[225,40,40]],t);}

// ---- metric definitions (radio = pick one path colouring) ----
const METRICS={
 "Doppler (Hz)":      {f:p=>p.d, col:v=>div(v,M.dopMax), lo:-M.dopMax,hi:M.dopMax, grad:"#2858eb,#ebebeb,#e12828",
   note:"red = approaching, blue = receding. Line of position is perpendicular to your path at each colour flip."},
 "Confidence (lock)": {f:p=>p.cf, col:v=>ramp(RYG,v/0.6), lo:0,hi:0.6, grad:"#c82828,#f0c846,#28a046",
   note:"matched-filter tone coherence. green = solid lock, red = unreliable. THIS is your data-trust map."},
 "Band level (dB)":   {f:p=>p.lv, col:v=>ramp(INF,(v-M.lvLo)/(M.lvHi-M.lvLo)), lo:M.lvLo,hi:M.lvHi, grad:"#0a051e,#bc3754,#f98e09,#fcffa4",
   note:"energy in the locked band = loudness of the beep tone."},
 "Input level (dB)":  {f:p=>p.il, col:v=>ramp(INF,(v-M.ilLo)/(M.ilHi-M.ilLo)), lo:M.ilLo,hi:M.ilHi, grad:"#0a051e,#bc3754,#f98e09,#fcffa4",
   note:"raw mic loudness (all sound). bright = wind / handling / traffic, often NOT the beep."},
 "Cadence conf":      {f:p=>p.cc, col:v=>ramp(RYG,v/0.5), lo:0,hi:0.5, grad:"#c82828,#f0c846,#28a046",
   note:"strength of the ~2.9/s beep rhythm. green = the beep pattern is clearly present."},
 "Frequency (Hz)":    {f:p=>p.fq, col:v=>div(v-M.fqMed,M.fqHalf), lo:M.fqMed-M.fqHalf,hi:M.fqMed+M.fqHalf, grad:"#2858eb,#ebebeb,#e12828",
   note:"absolute pitch (drift + Doppler). centred on session median "+M.fqMed+" Hz."},
 "Speed (m/s)":       {f:p=>p.sp, col:v=>ramp(VIR,v/M.spHi), lo:0,hi:M.spHi, grad:"#440154,#3b528b,#21918c,#5ec962,#fde725",
   note:"walking speed. Doppler is only meaningful while moving."},
};
function popup(p){return `<b>t=${p.t}s</b><br>Doppler ${p.d??'–'} Hz<br>conf ${p.cf??'–'}<br>band ${p.lv??'–'} dB<br>input ${p.il??'–'} dB<br>freq ${p.fq??'–'} Hz<br>cadence ${p.cd??'–'}/s (cc ${p.cc??'–'})<br>speed ${p.sp??'–'} m/s`;}

function buildLayer(name){const m=METRICS[name],lg=L.layerGroup();
 for(let i=1;i<P.length;i++){const a=P[i-1],b=P[i];const v=m.f(b);
  const op=Math.max(0.18,Math.min(1,(b.cf??0)*1.6));   // fade where lock is weak
  const color=(v==null)?"#888":m.col(v);
  L.polyline([[a.lat,a.lon],[b.lat,b.lon]],{color,weight:7,opacity:op,lineCap:"round"}).bindPopup(popup(b)).addTo(lg);
 } return lg;}

const baseColor={}; let first=null;
for(const name in METRICS){const lyr=buildLayer(name); baseColor[name]=lyr; if(!first){first=lyr; lyr.addTo(map);}}

// ---- overlays ----
function brg(a,b){const dLat=b.lat-a.lat,dLon=(b.lon-a.lon)*Math.cos(a.lat*Math.PI/180);return Math.atan2(dLon,dLat)*180/Math.PI;}
function arrowIcon(ang){return L.divIcon({className:'arw',iconSize:[28,28],iconAnchor:[14,14],
 html:`<svg width="28" height="28" viewBox="0 0 28 28" style="transform:rotate(${ang}deg)">
  <g fill="none" stroke="#fff" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round">
   <line x1="14" y1="25" x2="14" y2="7"/><polyline points="7,13 14,5 21,13"/></g>
  <g fill="none" stroke="#111" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
   <line x1="14" y1="25" x2="14" y2="7"/><polyline points="7,13 14,5 21,13"/></g></svg>`});}
// denser arrows as you zoom in
function arrowStep(z){return Math.max(2,Math.round(13*Math.pow(0.62,z-18)));}
const arrows=L.layerGroup().addTo(map);
function rebuildArrows(){arrows.clearLayers();const st=arrowStep(map.getZoom());
 for(let i=st;i<P.length;i+=st){const a=P[Math.max(0,i-1)],b=P[i];
  if(a.lat===b.lat&&a.lon===b.lon)continue;
  arrows.addLayer(L.marker([b.lat,b.lon],{interactive:false,icon:arrowIcon(brg(a,b))}));}}
rebuildArrows(); map.on("zoomend",rebuildArrows);
// raw (un-smoothed) GPS track, to see the canyon jitter
const rawTrack=L.polyline(P.map(p=>[p.rla,p.rlo]),{color:"#777",weight:1.4,opacity:.55,dashArray:"2 3"});

const lopL=L.layerGroup();
D.lops.forEach(l=>{L.polyline([l.a,l.b],{color:"#7a00cc",weight:2,dashArray:"5 5",opacity:.8}).bindTooltip("abeam t="+l.t+"s").addTo(lopL);});

// Doppler source-bearing field: each arrow points where Doppler says the source is.
// Where the source is real, arrows converge on it; multipath makes them scatter.
const fieldL=L.layerGroup();
function r2col(v){return v<=0?"#9aa1ab":ramp([[210,60,40],[240,200,70],[30,160,70]],Math.min(1,v/0.6));}
(D.field||[]).forEach(q=>{
  if(q.r2<0.15) return;                                   // hide incoherent patches (clutter)
  const Len=15, eLat=q.lat+(Len*q.uy)/D.mlat, eLon=q.lon+(Len*q.ux)/D.mlon;
  const col=r2col(q.r2), op=Math.max(0.4,Math.min(1,q.r2+0.3));
  L.polyline([[q.lat,q.lon],[eLat,eLon]],{color:col,weight:3,opacity:op})
   .bindTooltip(`this 6 m patch: Doppler points this way →<br>confidence (local R²)=${q.r2}, |u|=${q.mag}, ${q.n} samples`).addTo(fieldL);
  const ang=Math.atan2(q.ux,q.uy)*180/Math.PI;            // bearing of source-direction (0=N, CW)
  L.marker([eLat,eLon],{interactive:false,icon:L.divIcon({className:'arw',iconSize:[13,13],iconAnchor:[6,6],
    html:`<div style="transform:rotate(${ang}deg);color:${col};font-size:12px;text-shadow:0 0 2px #fff,0 0 2px #fff">▲</div>`})}).addTo(fieldL);
});

// per-sample Doppler arrows: red=approaching (forward), blue=receding (back), length = |shift|.
// Dense along the path, but only the ones within DQ_R px of the cursor are drawn ("flashlight").
const dqData=D.dquiver||[];
const dqL=L.layerGroup();
let dqActive=false, dqMouse=null, dqPending=false;
const DQ_R=85;   // cursor reveal radius (screen px)
function dqArrow(a){const col=a.d>0?"#d62728":"#1f77b4";
  L.polyline([[a.lat,a.lon],[a.ela,a.elo]],{color:col,weight:3,opacity:.95})
   .bindTooltip(`${a.d} Hz — ${a.d>0?"approaching (→)":"receding (←)"}`).addTo(dqL);
  const ang=Math.atan2((a.elo-a.lon)*Math.cos(a.lat*Math.PI/180),(a.ela-a.lat))*180/Math.PI;
  L.marker([a.ela,a.elo],{interactive:false,icon:L.divIcon({className:'arw',iconSize:[12,12],iconAnchor:[6,6],
    html:`<div style="transform:rotate(${ang}deg);color:${col};font-size:12px;text-shadow:0 0 2px #fff,0 0 2px #fff">▲</div>`})}).addTo(dqL);
}
function dqRender(){ dqL.clearLayers(); if(!dqActive||!dqMouse) return;
  for(const a of dqData){ const p=map.latLngToContainerPoint([a.lat,a.lon]);
    if(Math.hypot(p.x-dqMouse.x,p.y-dqMouse.y)<=DQ_R) dqArrow(a); } }
function dqSchedule(){ if(dqPending)return; dqPending=true; requestAnimationFrame(()=>{dqPending=false;dqRender();}); }
map.on("mousemove",e=>{ if(dqActive){ dqMouse=e.containerPoint; dqSchedule(); } });
map.on("mouseout",()=>{ if(dqActive){ dqMouse=null; dqL.clearLayers(); } });
map.on("move zoom",()=>{ if(dqActive) dqSchedule(); });

const markers=L.layerGroup();
L.marker(D.home).bindTooltip(D.homeLabel||"HOME",{permanent:true,direction:"top"}).addTo(markers);
L.circleMarker(D.src,{radius:9,color:"#000",weight:2,fillColor:"#39ff14",fillOpacity:1})
  .bindTooltip(`Doppler source fit (R²=${D.R2}, ${D.verdict})`,{permanent:false}).addTo(markers);
markers.addTo(map);

L.control.layers(baseColor,{"↦ Doppler arrows (red=toward, blue=away)":dqL,"↗ Doppler → source (bearing field)":fieldL,
  "➤ walk direction (denser on zoom)":arrows,"⟂ abeam lines-of-position":lopL,"· raw GPS (jittery)":rawTrack,
  "⚑ home + source fit":markers},{collapsed:false,position:"topright"}).addTo(map);

// ---- legend (updates with active colouring) ----
const lg=L.control({position:"bottomleft"});
lg.onAdd=()=>{const d=L.DomUtil.create("div","lg panel");d.id="leg";return d;};
lg.addTo(map);
function setLegend(name){const m=METRICS[name];
 document.getElementById("leg").innerHTML=
  `<b>${name}</b><div class="bar" style="--g:${m.grad}"></div>`+
  `<div class="bl"><span>${m.lo}</span><span>${m.hi}</span></div>`+
  `<div class="note">${m.note}</div>`+
  `<div class="note">Source fit: R²=${D.R2}, ${D.dh} m from home — <b>${D.verdict}</b>.<br>Opacity ∝ lock confidence (faint = don't trust). Track is GPS-smoothed (1.5s); toggle "raw GPS" to see canyon jitter.</div>`;}
let active="Doppler (Hz)"; setLegend(active);
map.on("baselayerchange",e=>{active=e.name;setLegend(e.name);});

// legend for the Doppler bearing-field overlay (appears only when it's toggled on)
const fleg=L.control({position:"bottomright"});
fleg.onAdd=()=>{const d=L.DomUtil.create("div","lg panel");d.id="fieldleg";d.style.display="none";
 d.innerHTML=`<b>↗ Doppler → source</b><br>each arrow = one ~6 m patch of path; it points where`+
  ` that patch's Doppler (from your passes through it) says the source is.`+
  `<div class="bar" style="--g:#d23c28,#f0c846,#1ea046"></div><div class="bl"><span>noisy R²≤0</span><span>coherent R²≥0.6</span></div>`+
  `<div class="note">Best viewed zoomed OUT. Arrows that <b>converge</b> on a spot ⇒ likely the source; arrows that <b>scatter</b> ⇒ canyon echo. Not per-cm: a direction needs many passes, so it's one arrow per patch.</div>`;return d;};
fleg.addTo(map);
map.on("overlayadd",e=>{if(e.name.indexOf("Doppler →")>=0)document.getElementById("fieldleg").style.display="block";});
map.on("overlayremove",e=>{if(e.name.indexOf("Doppler →")>=0)document.getElementById("fieldleg").style.display="none";});

// legend for the per-sample Doppler arrows
const dqleg=L.control({position:"bottomright"});
dqleg.onAdd=()=>{const d=L.DomUtil.create("div","lg panel");d.id="dqleg";d.style.display="none";
 d.innerHTML=`<b>↦ Doppler arrows</b><br><span style="color:#d62728;font-weight:700">red →</span> approaching (source ahead this way)<br>`+
  `<span style="color:#1f77b4;font-weight:700">blue →</span> receding (source behind you)`+
  `<div class="note"><b>Move your cursor over the path</b> to reveal nearby arrows. Length = how fast the distance is changing; they shrink & flip red↔blue at closest approach. Per GPS sample — points only along your path, never sideways.</div>`;return d;};
dqleg.addTo(map);
map.on("overlayadd",e=>{if(e.name.indexOf("Doppler arrows")>=0){document.getElementById("dqleg").style.display="block";dqActive=true;}});
map.on("overlayremove",e=>{if(e.name.indexOf("Doppler arrows")>=0){document.getElementById("dqleg").style.display="none";dqActive=false;dqL.clearLayers();}});

// ---- measurement (walk) selector: switches by loading that walk's page ----
if(D.walks && D.walks.length>1){
  const sc=L.control({position:"topleft"});
  sc.onAdd=()=>{const d=L.DomUtil.create("div","lg panel");
    const opts=D.walks.map(w=>`<option value="${w.href}"${w.id===D.current?" selected":""}>${w.label}</option>`).join("");
    d.innerHTML=`<b>measurement</b><br><select id="walkSel" style="margin-top:4px;font:12px system-ui;max-width:210px">${opts}</select>`;
    L.DomEvent.disableClickPropagation(d); L.DomEvent.disableScrollPropagation(d);
    d.querySelector("#walkSel").onchange=e=>{location.href=e.target.value;};
    return d;};
  sc.addTo(map);
}

// ---- fit EXACTLY to data extent (no tile slack) ----
map.fitBounds(D.bbox,{padding:[8,8]});
</script></body></html>"""

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
