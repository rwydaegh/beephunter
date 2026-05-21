"""Rigorous point-source Doppler localization + interactive HTML walk map.

Point-source model (uses EVERY sample, not just zero-crossings):
    doppler_i = (f0/c) * v_i . (S - p_i)/|S - p_i|
Fit the 2-D source position S by nonlinear least squares (multi-start), bootstrap
for uncertainty, and report how much of the Doppler it actually explains (R²).
Then write walk_map.html: Leaflet, tight fitBounds, Doppler-colored path with
direction arrows, home + source markers.
"""
import glob, json, math, os, sys
import numpy as np
from scipy.optimize import least_squares
from bh_config import HOME
DL=os.path.expanduser(r"~/Downloads"); C=343.0
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime,reverse=True)[0]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {"t":g("tw")/1000,"f":g("f"),"lvl":g("lvl"),"lat":g("lat"),"lon":g("lon"),"spd":g("spd")}
def sm(x,k):
    o=np.copy(x)
    for i in range(len(x)): s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.mean(s) if len(s) else np.nan
    return o
def mf(x,k):
    o=np.copy(x)
    for i in range(len(x)): s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.median(s) if len(s) else np.nan
    return o

def main():
    P=load(newest("hunter")); R=load(newest("reference"))
    dt=0.2; t0=max(P["t"][0],R["t"][0]); t1=min(P["t"][-1],R["t"][-1]); g=np.arange(0,t1-t0,dt)
    pf=np.interp(g,P["t"]-t0,P["f"]); rf=sm(np.interp(g,R["t"]-t0,R["f"]),int(8/dt)); f0=np.nanmedian(rf)
    lat=np.interp(g,P["t"]-t0,P["lat"]); lon=np.interp(g,P["t"]-t0,P["lon"]); spd=np.interp(g,P["t"]-t0,np.nan_to_num(P["spd"],nan=0))
    lvl=np.interp(g,P["t"]-t0,P["lvl"])
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat.mean()))
    X=(lon-lon.mean())*mlon; Y=(lat-lat.mean())*mlat
    Xs,Ys=sm(X,int(1/dt)),sm(Y,int(1/dt)); vx,vy=np.gradient(Xs,dt),np.gradient(Ys,dt); sp=np.hypot(vx,vy)
    dop=pf-rf; vmax=1.6*np.maximum(spd,.3)*f0/C+2.0; dop=np.where(np.abs(dop)<=vmax,dop,np.nan); dop=sm(mf(dop,3),int(1.2/dt))
    m=np.isfinite(dop)&(sp>0.6)
    Xm,Ym,vxm,vym,dm=X[m],Y[m],vx[m],vy[m],dop[m]

    def resid(S):
        dx=S[0]-Xm; dy=S[1]-Ym; r=np.hypot(dx,dy)+1e-6
        pred=(f0/C)*(vxm*dx+vym*dy)/r
        return pred-dm
    best=None
    for gx in range(-60,61,20):
        for gy in range(-60,61,20):
            try:
                r=least_squares(resid,[gx,gy],method="lm",max_nfev=2000)
                if best is None or r.cost<best.cost: best=r
            except Exception: pass
    S=best.x
    res=resid(S); rms=np.sqrt(np.mean(res**2)); R2=1-np.sum(res**2)/np.sum((dm-dm.mean())**2)
    # bootstrap uncertainty
    boot=[]
    rng=np.random.default_rng(0)
    for _ in range(200):
        idx=rng.integers(0,len(dm),len(dm))
        def rb(S):
            dx=S[0]-Xm[idx]; dy=S[1]-Ym[idx]; r=np.hypot(dx,dy)+1e-6
            return (f0/C)*(vxm[idx]*dx+vym[idx]*dy)/r-dm[idx]
        try: boot.append(least_squares(rb,S,method="lm",max_nfev=800).x)
        except Exception: pass
    boot=np.array(boot); sx,sy=(boot.std(0) if len(boot)>10 else (np.nan,np.nan))
    slat=lat.mean()+S[1]/mlat; slon=lon.mean()+S[0]/mlon
    dh=math.hypot((slat-HOME[0])*mlat,(slon-HOME[1])*mlon)
    print(f"point-source fit: S=({S[0]:+.0f},{S[1]:+.0f})m -> {slat:.6f},{slon:.6f}")
    print(f"  RMS residual {rms:.2f} Hz vs raw Doppler std {dm.std():.2f} Hz | R²={R2:.2f}")
    print(f"  bootstrap uncertainty: ±({sx:.0f}, {sy:.0f}) m | {dh:.0f} m from home")
    verdict = "TRUSTWORTHY" if (R2>0.5 and max(sx,sy)<15) else ("WEAK" if R2>0.25 else "NOT SUPPORTED (multipath dominates)")
    print(f"  verdict: {verdict}")

    # ---- HTML ----
    step=max(1,len(g)//500)
    pts=[{"lat":round(lat[i],7),"lon":round(lon[i],7),"d":round(float(np.nan_to_num(dop[i])),2)} for i in range(0,len(g),step)]
    data={"pts":pts,"home":HOME,"src":[slat,slon],"su":[float(sx) if np.isfinite(sx) else 0,float(sy) if np.isfinite(sy) else 0],
          "R2":round(R2,2),"rms":round(rms,2),"f0":round(f0,1),"verdict":verdict}
    html=TEMPLATE.replace("__DATA__",json.dumps(data))
    open(os.path.join(ROOT,"walk_map.html"),"w",encoding="utf-8").write(html); print("saved walk_map.html")

TEMPLATE=r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>BeepHunter walk</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0}.lg{background:#fff;padding:8px 10px;border-radius:6px;font:12px sans-serif;line-height:1.5;box-shadow:0 1px 5px rgba(0,0,0,.4)}.arw div{font-size:16px;line-height:16px}</style>
</head><body><div id="map"></div><script>
const D=__DATA__;
const map=L.map('map',{maxZoom:21});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxNativeZoom:19,maxZoom:21,attribution:'© OpenStreetMap'}).addTo(map);
const P=D.pts;
function col(v){let f=Math.max(0,Math.min(1,(v+8)/16));return `rgb(${Math.round(255*f)},${Math.round(60*(1-Math.abs(f-.5)*2))},${Math.round(255*(1-f))})`;}
const bounds=[];
for(let i=1;i<P.length;i++){
  L.polyline([[P[i-1].lat,P[i-1].lon],[P[i].lat,P[i].lon]],{color:col(P[i].d),weight:6,opacity:.95}).addTo(map);
  bounds.push([P[i].lat,P[i].lon]);
}
// direction arrows every N points
function brg(a,b){const dLat=b.lat-a.lat,dLon=(b.lon-a.lon)*Math.cos(a.lat*Math.PI/180);return Math.atan2(dLon,dLat)*180/Math.PI;}
for(let i=8;i<P.length;i+=8){
  const ang=brg(P[i-1],P[i]);
  const ic=L.divIcon({className:'arw',html:`<div style="transform:rotate(${ang-90}deg)">▶</div>`,iconSize:[16,16],iconAnchor:[8,8]});
  L.marker([P[i].lat,P[i].lon],{icon:ic,interactive:false}).addTo(map);
}
L.marker(D.home).addTo(map).bindTooltip('HOME #24',{permanent:true});
const src=L.circleMarker(D.src,{radius:9,color:'#000',weight:2,fillColor:'lime',fillOpacity:1}).addTo(map).bindTooltip('Doppler point-source est',{permanent:true});
// uncertainty ellipse (approx as circle of mean sigma in meters -> deg)
const su=Math.max(D.su[0],D.su[1]); if(su>0){const r=su; L.circle(D.src,{radius:r,color:'lime',fill:false,dashArray:'4 4'}).addTo(map);}
map.fitBounds(bounds,{padding:[30,30]});
const lg=L.control({position:'topright'});lg.onAdd=()=>{const d=L.DomUtil.create('div','lg');
 d.innerHTML=`<b>BeepHunter walk</b><br>arrows = walk direction<br>🔴 approaching · 🔵 receding<br>★ source fit: R²=${D.R2}, ±${Math.round(su)}m<br><b>${D.verdict}</b>`;return d;};
lg.addTo(map);
</script></body></html>"""

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
