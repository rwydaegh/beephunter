"""Per-sample Doppler arrows (the user's idea): at points spaced along the walk,
draw   arrow = sign(Doppler) * heading_unit_vector,  length proportional to |Doppler|.

Red  = approaching -> arrow points FORWARD along the path (source is ahead this way).
Blue = receding    -> arrow points BACKWARD            (source is behind, back that way).
Arrows shrink to zero and flip at closest approach (source is then straight out to the side).
No binning. Tight-cropped OSM basemap. Big png.
"""
import glob, io, json, math, os, sys, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from bh_config import HOME, CONTACT

DL=os.path.expanduser(r"~/Downloads"); C=343.0; Z=19
UA={"User-Agent":f"beephunter-dev/1.0 ({CONTACT})"}
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime)[-1]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {k:g(k) for k in ["tw","f","conf","lat","lon","spd"]}
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
    gr=np.arange(0,rt1-rt0,dt); rf=np.interp(g+(t0-rt0),gr,sm(np.interp(gr,R["tw"]/1000-rt0,R["f"]),int(8/dt))); f0=np.nanmedian(rf)
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(np.nanmean(lat)))
    X=(lon-lon.mean())*mlon; Y=(lat-lat.mean())*mlat
    Xs,Ys=sm(X,int(1/dt)),sm(Y,int(1/dt)); vx,vy=np.gradient(Xs,dt),np.gradient(Ys,dt); sp=np.hypot(vx,vy)
    dop=pf-rf; vmax=1.6*np.maximum(spd,.3)*f0/C+2.0; dop=np.where(np.abs(dop)<=vmax,dop,np.nan); dop=sm(mf(dop,3),int(1.2/dt))

    # sample arrows every ~ARC metres along the smoothed path, where moving & locked
    XsS=(lonS-lon.mean())*mlon; YsS=(latS-lat.mean())*mlat
    arc=np.concatenate([[0],np.cumsum(np.hypot(np.diff(XsS),np.diff(YsS)))])
    ARC=3.5; picks=[]; nexts=0.0
    for i in range(len(g)):
        if arc[i]>=nexts and np.isfinite(dop[i]) and sp[i]>0.6 and conf[i]>0.30 and sp[i]>1e-3:
            picks.append(i); nexts=arc[i]+ARC
    picks=np.array(picks)
    dm=np.nanpercentile(np.abs(dop[np.isfinite(dop)]),95); Lmax=11.0   # metres for a max-Doppler arrow

    # arrow endpoints (meters): vec = heading_unit * (dop/dm) * Lmax  (sign handled by dop)
    ends=[]
    for i in picks:
        hx_,hy_=vx[i]/sp[i],vy[i]/sp[i]; s=np.clip(dop[i]/dm,-1.4,1.4)*Lmax
        ends.append((latS[i],lonS[i],latS[i]+(hy_*s)/mlat,lonS[i]+(hx_*s)/mlon,dop[i]))

    # ---- tight bbox over path + arrow tips + home ----
    glat=list(latS)+[HOME[0]]+[e[2] for e in ends]; glon=list(lonS)+[HOME[1]]+[e[3] for e in ends]
    mar=4.0; la0,la1=min(glat)-mar/mlat,max(glat)+mar/mlat; lo0,lo1=min(glon)-mar/mlon,max(glon)+mar/mlon
    x0,y0=[int(v) for v in deg2num(la1,lo0,Z)]; x1,y1=[int(v) for v in deg2num(la0,lo1,Z)]
    W,H=(x1-x0+1)*256,(y1-y0+1)*256; cv=Image.new("RGB",(W,H))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try: cv.paste(tile(Z,tx,ty),((tx-x0)*256,(ty-y0)*256))
            except Exception as e: print("tile",e)
    px=lambda la,lo:((deg2num(la,lo,Z)[0]-x0)*256,(deg2num(la,lo,Z)[1]-y0)*256)
    left,top=px(la1,lo0); right,bottom=px(la0,lo1)

    fig,ax=plt.subplots(figsize=((right-left)/45,(bottom-top)/45),dpi=300); ax.imshow(np.asarray(cv),origin="upper")
    # faint path for context
    Pp=np.array([px(latS[i],lonS[i]) for i in range(len(g))])
    ax.plot(Pp[:,0],Pp[:,1],"-",color="#555",lw=1.2,alpha=.5,zorder=3)
    # per-sample Doppler arrows: red = approaching (forward), blue = receding (backward)
    for (la,lo,ela,elo,d) in ends:
        sx,sy=px(la,lo); ex,ey=px(ela,elo); col="#d62728" if d>0 else "#1f77b4"
        ax.annotate("",xy=(ex,ey),xytext=(sx,sy),zorder=5,
                    arrowprops=dict(arrowstyle="-|>",color=col,lw=2.0,shrinkA=0,shrinkB=0,
                                    mutation_scale=13,path_effects=[pe.withStroke(linewidth=3,foreground="white")]))
    hx,hy=px(*HOME); ax.plot(hx,hy,"s",ms=14,color="#111",mec="white",mew=1.6,zorder=7)
    ax.annotate("HOME",(hx,hy),color="#111",fontsize=11,fontweight="bold",xytext=(8,7),textcoords="offset points",
                zorder=8,path_effects=[pe.withStroke(linewidth=3,foreground="white")])
    ax.set_xlim(left,right); ax.set_ylim(bottom,top); ax.axis("off")
    fig.savefig(os.path.join(ROOT,"doppler_arrows.png"),dpi=300,bbox_inches="tight",pad_inches=0.0)
    print(f"saved doppler_arrows.png  ({len(ends)} arrows, every {ARC:.1f} m; max arrow = {dm:.1f} Hz)")

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
