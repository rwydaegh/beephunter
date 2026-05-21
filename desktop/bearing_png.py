"""Doppler source-bearing field as a TIGHT png (shrinkwrapped to drawn data).

Renders the walk path colored by Doppler, the per-6m-cell bearing arrows
(big heads) pointing where Doppler says the source is, the least-squares
consensus intersection, and home. The OSM basemap is cropped to the exact
extent of everything drawn (no tile-boundary padding).
"""
import glob, io, json, math, os, sys, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from bh_config import HOME, CONTACT

DL=os.path.expanduser(r"~/Downloads"); C=343.0; Z=19
UA={"User-Agent":f"beephunter-dev/1.0 ({CONTACT})"}
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime)[-1]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {k:g(k) for k in ["tw","f","conf","lvl","lat","lon","spd"]}
def sm(x,k):
    o=np.copy(x)
    for i in range(len(x)): s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.mean(s) if len(s) else np.nan
    return o
def mf(x,k):
    o=np.copy(x)
    for i in range(len(x)): s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.median(s) if len(s) else np.nan
    return o
def deg2num(lat,lon,z): n=2**z; return (lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n
def tile(z,x,y): return Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(
    f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",headers=UA),timeout=20).read())).convert("RGB")

def main():
    R=load(newest("reference")); rt0,rt1=R["tw"][0]/1000,R["tw"][-1]/1000
    Pf=None
    for h in sorted(glob.glob(os.path.join(DL,"beephunter_hunter_*.json")),key=os.path.getmtime,reverse=True):
        tw=np.array([x.get("tw") for x in json.load(open(h,encoding="utf-8"))["rows"]],float)/1000
        if np.isfinite(tw).any() and tw[np.isfinite(tw)][-1]>rt0 and tw[np.isfinite(tw)][0]<rt1: Pf=h; break
    P=load(Pf)
    dt=0.2; t0=max(P["tw"][0],R["tw"][0])/1000; t1=min(P["tw"][-1],R["tw"][-1])/1000; g=np.arange(0,t1-t0,dt)
    I=lambda s,k:np.interp(g,s["tw"]/1000-t0,s[k])
    pf=I(P,"f"); spd=np.interp(g,P["tw"]/1000-t0,np.nan_to_num(P["spd"],nan=0)); conf=I(P,"conf")
    lat=I(P,"lat"); lon=I(P,"lon"); latS=sm(lat,int(1.5/dt)); lonS=sm(lon,int(1.5/dt))
    gr=np.arange(0,rt1-rt0,dt); rf_full=sm(np.interp(gr,R["tw"]/1000-rt0,R["f"]),int(8/dt))
    rf=np.interp(g+(t0-rt0),gr,rf_full); f0=np.nanmedian(rf)
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(np.nanmean(lat)))
    X=(lon-lon.mean())*mlon; Y=(lat-lat.mean())*mlat
    Xs,Ys=sm(X,int(1/dt)),sm(Y,int(1/dt)); vx,vy=np.gradient(Xs,dt),np.gradient(Ys,dt); sp=np.hypot(vx,vy)
    dop=pf-rf; vmax=1.6*np.maximum(spd,.3)*f0/C+2.0; dop=np.where(np.abs(dop)<=vmax,dop,np.nan); dop=sm(mf(dop,3),int(1.2/dt))
    m=np.isfinite(dop)&(sp>0.6)&(conf>0.30)

    # bearing field
    xs,ys,vxs,vys,ds=X[m],Y[m],vx[m],vy[m],dop[m]; cell=6.0; field=[]
    xe=np.arange(math.floor(xs.min()/cell)*cell,xs.max()+cell,cell)
    ye=np.arange(math.floor(ys.min()/cell)*cell,ys.max()+cell,cell)
    for xi in xe:
        for yi in ye:
            c=(xs>=xi)&(xs<xi+cell)&(ys>=yi)&(ys<yi+cell)
            if c.sum()<6: continue
            A=np.column_stack([vxs[c],vys[c]]); sv=np.linalg.svd(A,compute_uv=False)
            if sv[0]<1e-6 or sv[-1]<0.30*sv[0]: continue
            b=ds[c]*C/f0; u,*_=np.linalg.lstsq(A,b,rcond=None); mag=math.hypot(*u)
            if mag<1e-3: continue
            pred=A@u; sstot=np.sum((b-b.mean())**2); r2c=1-np.sum((b-pred)**2)/sstot if sstot>1e-9 else 0.0
            field.append(dict(cx=xi+cell/2,cy=yi+cell/2,ux=u[0]/mag,uy=u[1]/mag,mag=mag,r2=r2c))
    conf_cells=[q for q in field if q["r2"]>0.3]
    # LS ray intersection (consensus) in X/Y meters
    M=np.zeros((2,2)); v=np.zeros(2)
    for q in conf_cells:
        p=np.array([q["cx"],q["cy"]]); d=np.array([q["ux"],q["uy"]]); Pm=np.eye(2)-np.outer(d,d); M+=Pm; v+=Pm@p
    cons=np.linalg.solve(M,v) if len(conf_cells)>=2 else None

    # geo helpers
    hX=(HOME[1]-lon.mean())*mlon; hY=(HOME[0]-lat.mean())*mlat
    def XYtoll(x,y): return lat.mean()+y/mlat, lon.mean()+x/mlon
    # ---- collect everything drawn for tight bbox ----
    geo=[(latS[i],lonS[i]) for i in range(len(g))]+[HOME]
    for q in conf_cells:
        la,lo=XYtoll(q["cx"],q["cy"]); geo.append((la,lo))
        ela,elo=XYtoll(q["cx"]+15*q["ux"],q["cy"]+15*q["uy"]); geo.append((ela,elo))
    if cons is not None: geo.append(XYtoll(*cons))
    glat=np.array([p[0] for p in geo]); glon=np.array([p[1] for p in geo])
    mar=4.0; latm=mar/mlat; lonm=mar/mlon
    la0,la1=glat.min()-latm,glat.max()+latm; lo0,lo1=glon.min()-lonm,glon.max()+lonm

    # ---- stitch + tight crop ----
    x0,y0=[int(v) for v in deg2num(la1,lo0,Z)]; x1,y1=[int(v) for v in deg2num(la0,lo1,Z)]
    W,H=(x1-x0+1)*256,(y1-y0+1)*256; cv=Image.new("RGB",(W,H))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try: cv.paste(tile(Z,tx,ty),((tx-x0)*256,(ty-y0)*256))
            except Exception as e: print("tile",e)
    px=lambda la,lo:((deg2num(la,lo,Z)[0]-x0)*256,(deg2num(la,lo,Z)[1]-y0)*256)
    left,top=px(la1,lo0); right,bottom=px(la0,lo1)

    import matplotlib.patheffects as pe
    halo=[pe.withStroke(linewidth=3,foreground="white")]
    fig,ax=plt.subplots(figsize=((right-left)/60,(bottom-top)/45),dpi=300); ax.imshow(np.asarray(cv),origin="upper")
    # path colored by Doppler (the hero layer)
    Pp=np.array([px(latS[i],lonS[i]) for i in range(len(g))])
    segs=np.stack([Pp[:-1],Pp[1:]],axis=1); dv=np.nan_to_num(dop[1:]); dm=np.nanpercentile(np.abs(dop[np.isfinite(dop)]),95)
    lc=LineCollection(segs,cmap="coolwarm",norm=plt.Normalize(-dm,dm),linewidths=5,alpha=.95,
                      path_effects=[pe.Stroke(linewidth=7,foreground="white"),pe.Normal()]); lc.set_array(dv); ax.add_collection(lc)
    # bearing rays: thin, indigo, opacity = confidence; modest visible head; converge => source
    for q in conf_cells:
        la,lo=XYtoll(q["cx"],q["cy"]); ela,elo=XYtoll(q["cx"]+22*q["ux"],q["cy"]+22*q["uy"])
        sx,sy=px(la,lo); ex,ey=px(ela,elo); op=0.45+0.5*min(1,q["r2"]/0.6)
        ax.annotate("",xy=(ex,ey),xytext=(sx,sy),zorder=5,
                    arrowprops=dict(arrowstyle="-|>",color="#3b0a6b",lw=1.8,alpha=op,shrinkA=0,shrinkB=0,
                                    mutation_scale=15,path_effects=[pe.withStroke(linewidth=3.2,foreground="white")]))
        ax.plot(sx,sy,"o",ms=3.2,color="#3b0a6b",alpha=op,zorder=5)
    # home only (Doppler-consensus star, title, and colorbar removed per request)
    hx,hy=px(*HOME); ax.plot(hx,hy,"s",ms=14,color="#111",mec="white",mew=1.6,zorder=7)
    ax.annotate("HOME",(hx,hy),color="#111",fontsize=11,fontweight="bold",xytext=(8,7),textcoords="offset points",zorder=8,path_effects=halo)
    ax.set_xlim(left,right); ax.set_ylim(bottom,top); ax.axis("off")
    fig.savefig(os.path.join(ROOT,"bearing_field.png"),dpi=300,bbox_inches="tight",pad_inches=0.0); print("saved bearing_field.png")
    if cons is not None:
        print(f"consensus at ({cons[0]:+.0f}E,{cons[1]:+.0f}N) m from home, dist {math.hypot(cons[0]-hX*0+cons[0]-0,0)*0+math.hypot(cons[0]- (0),cons[1]-(0)):.0f} m"
              if False else f"consensus ({cons[0]:+.0f}E,{cons[1]:+.0f}N)m from walk-center; home at ({hX:+.0f},{hY:+.0f})m")

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
