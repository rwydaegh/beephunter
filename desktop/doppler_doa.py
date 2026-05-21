"""Decisive test: is the Doppler a localizable POINT source, or sound arriving
from a fixed DIRECTION (guided along the canyon)?

Model: f_doppler = (f0/c) * v . u, where v is your velocity vector and u is the
unit vector pointing toward the wavefront's source. For a point source, u
rotates as you move (so a single constant u fits poorly). For a distant/guided
source, u is ~constant -> a single direction fits well, and its bearing tells us
which way the sound comes from (but not the distance).
"""
import glob, json, math, os, sys
import numpy as np
from bh_config import HOME
DL=os.path.expanduser(r"~/Downloads"); C=343.0
def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime,reverse=True)[0]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {"t":g("tw")/1000,"f":g("f"),"lat":g("lat"),"lon":g("lon"),"spd":g("spd")}
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
    spd=np.interp(g,P["t"]-t0,np.nan_to_num(P["spd"],nan=0)); lat=np.interp(g,P["t"]-t0,P["lat"]); lon=np.interp(g,P["t"]-t0,P["lon"])
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat.mean()))
    X=(lon-lon.mean())*mlon; Y=(lat-lat.mean())*mlat
    Xs,Ys=sm(X,int(1/dt)),sm(Y,int(1/dt)); vx,vy=np.gradient(Xs,dt),np.gradient(Ys,dt)
    dop=pf-rf; vmax=1.6*np.maximum(spd,.3)*f0/C+2.0; dop=np.where(np.abs(dop)<=vmax,dop,np.nan); dop=sm(mf(dop,3),int(1.2/dt))
    sp=np.hypot(vx,vy)
    m=np.isfinite(dop)&(sp>0.6)
    # LS: dop = (f0/c)*(vx*ux+vy*uy)
    A=np.column_stack([vx[m],vy[m]])*(f0/C); b=dop[m]
    u,res,rank,sv=np.linalg.lstsq(A,b,rcond=None)
    pred=A@u; ss_res=np.sum((b-pred)**2); ss_tot=np.sum((b-b.mean())**2); R2=1-ss_res/ss_tot
    bearing=(math.degrees(math.atan2(u[0],u[1]))+360)%360  # 0=N,90=E
    print(f"samples used: {m.sum()} | f0={f0:.1f} Hz")
    print(f"plane-wave fit: |u|={np.hypot(*u):.2f} (≈1 ⇒ single plane wave)  R²={R2:.2f}")
    print(f"  => sound arrives from bearing {bearing:.0f}° (0=N,90=E,180=S,270=W)")
    # canyon axis from track principal direction
    pts=np.column_stack([X[m],Y[m]]); pts=pts-pts.mean(0); _,_,V=np.linalg.svd(pts,full_matrices=False)
    axis=V[0]; axis_bear=(math.degrees(math.atan2(axis[0],axis[1]))+360)%360
    print(f"canyon/walk axis bearing ≈ {axis_bear:.0f}° (and {(axis_bear+180)%360:.0f}°)")
    # sliding-window bearing stability (point source -> rotates; plane wave -> steady)
    print("sliding-window DOA bearing (30s windows):")
    w=int(30/dt)
    for s in range(0,len(g)-w,w//2):
        mm=m.copy(); mm[:s]=False; mm[s+w:]=False
        if mm.sum()<20: continue
        Aw=np.column_stack([vx[mm],vy[mm]])*(f0/C); bw=dop[mm]
        uu,_,_,_=np.linalg.lstsq(Aw,bw,rcond=None); br=(math.degrees(math.atan2(uu[0],uu[1]))+360)%360
        print(f"  t={g[s]:.0f}-{g[s+w]:.0f}s: bearing {br:.0f}°  |u|={np.hypot(*uu):.2f}")
    # where does that bearing point relative to home?
    dN=(HOME[0]-lat.mean())*mlat; dE=(HOME[1]-lon.mean())*mlon
    home_bear=(math.degrees(math.atan2(dE,dN))+360)%360
    print(f"home is at bearing {home_bear:.0f}° from walk centroid, {math.hypot(dN,dE):.0f} m away")

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
