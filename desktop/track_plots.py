"""Standalone track plots on OSM: Doppler, raw amplitude, differential amplitude,
plus a spatio-temporal amplitude analysis (level vs time and vs space)."""
import io, json, glob, math, os, sys, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bh_config import HOME, CONTACT
DL = os.path.expanduser(r"~/Downloads"); UA={"User-Agent":f"beephunter-dev/1.0 ({CONTACT})"}
C=343.0; Z=19
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime,reverse=True)[0]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {"t":g("tw")/1000,"f":g("f"),"lvl":g("lvl"),"inlvl":g("inlvl"),"lat":g("lat"),"lon":g("lon"),"spd":g("spd")}
def sm(x,k):
    o=np.copy(x)
    for i in range(len(x)):
        s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.mean(s) if len(s) else np.nan
    return o
def mf(x,k):
    o=np.copy(x)
    for i in range(len(x)):
        s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.median(s) if len(s) else np.nan
    return o
def deg2num(lat,lon,z): n=2**z; return (lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n
def tile(z,x,y): return Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",headers=UA),timeout=20).read())).convert("RGB")

def basemap(lat,lon):
    lats=list(lat)+[HOME[0]]; lons=list(lon)+[HOME[1]]
    la0,la1=min(lats)-.00018,max(lats)+.00018; lo0,lo1=min(lons)-.0003,max(lons)+.0003
    x0,y0=[int(v) for v in deg2num(la1,lo0,Z)]; x1,y1=[int(v) for v in deg2num(la0,lo1,Z)]
    W,H=(x1-x0+1)*256,(y1-y0+1)*256; cv=Image.new("RGB",(W,H))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try: cv.paste(tile(Z,tx,ty),((tx-x0)*256,(ty-y0)*256))
            except Exception as e: print("tile",e)
    px=lambda la,lo:((deg2num(la,lo,Z)[0]-x0)*256,(deg2num(la,lo,Z)[1]-y0)*256)
    return np.asarray(cv),W,H,px

def track_fig(lat,lon,vals,cmap,vlim,title,fname,cbar,markpts=None):
    img,W,H,px=basemap(lat,lon)
    fig,ax=plt.subplots(figsize=(W/110,H/110),dpi=130); ax.imshow(img,origin="upper")
    P=[px(lat[i],lon[i]) for i in range(len(lat))]
    for i in range(1,len(P)):
        v=vals[i]
        if not np.isfinite(v): col=(.5,.5,.5,1)
        else: col=cmap((np.clip(v,vlim[0],vlim[1])-vlim[0])/(vlim[1]-vlim[0]))
        ax.plot([P[i-1][0],P[i][0]],[P[i-1][1],P[i][1]],"-",color=col,lw=5,solid_capstyle="round")
    hx,hy=px(*HOME); ax.plot(hx,hy,"*",ms=26,color="#19d219",mec="k",mew=1.3)
    ax.annotate("HOME #24",(hx,hy),color="#0a6b0a",fontsize=12,fontweight="bold",xytext=(8,-13),textcoords="offset points")
    if markpts:
        for la,lo in markpts: mx,my=px(la,lo); ax.plot(mx,my,"o",ms=9,mfc="yellow",mec="k")
    sm_=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(*vlim)); sm_.set_array([])
    plt.colorbar(sm_,ax=ax,label=cbar,shrink=.7)
    ax.set_title(title,fontsize=12,fontweight="bold"); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT,fname),dpi=130,bbox_inches="tight"); print("saved",fname)

def main():
    P=load(newest("hunter")); R=load(newest("reference"))
    dt=0.2; t0=max(P["t"][0],R["t"][0]); t1=min(P["t"][-1],R["t"][-1]); g=np.arange(0,t1-t0,dt)
    pf=np.interp(g,P["t"]-t0,P["f"]); rf=sm(np.interp(g,R["t"]-t0,R["f"]),int(8/dt))
    plv=np.interp(g,P["t"]-t0,P["lvl"]); rlv=np.interp(g,R["t"]-t0,R["lvl"])
    spd=np.interp(g,P["t"]-t0,np.nan_to_num(P["spd"],nan=0)); lat=np.interp(g,P["t"]-t0,P["lat"]); lon=np.interp(g,P["t"]-t0,P["lon"])
    f0=np.nanmedian(rf)
    dop=pf-rf
    vmax=1.6*np.maximum(spd,.3)*f0/C+2.0; dop=np.where(np.abs(dop)<=vmax,dop,np.nan); dop=sm(mf(dop,3),int(1.2/dt))
    difflv=plv-rlv

    # 1) Doppler track
    track_fig(lat,lon,dop,plt.cm.coolwarm,(-8,8),
              "Doppler over the walk (red = approaching, blue = receding)","track_doppler.png","Doppler shift (Hz)")
    # 2) raw amplitude track
    a_lo,a_hi=np.nanpercentile(plv,5),np.nanpercentile(plv,95)
    track_fig(lat,lon,sm(plv,int(1.0/dt)),plt.cm.inferno,(a_lo,a_hi),
              "Band level over the walk (raw amplitude)","track_amplitude.png","level (dB)")
    # 3) differential amplitude (source temporal modulation removed via reference)
    d_lo,d_hi=np.nanpercentile(difflv,5),np.nanpercentile(difflv,95)
    track_fig(lat,lon,sm(difflv,int(1.0/dt)),plt.cm.viridis,(d_lo,d_hi),
              "Differential level (hunter − reference): source modulation removed","track_diffamp.png","Δlevel (dB)")

    # 4) spatio-temporal amplitude analysis figure
    fig,ax=plt.subplots(2,1,figsize=(13,8)); fig.suptitle("Spatio-temporal amplitude analysis",fontweight="bold")
    a=ax[0]; a.plot(g,sm(plv,int(.6/dt)),color="#1f77b4",label="hunter level"); a.plot(g,sm(rlv,int(.6/dt)),color="#d62728",label="reference level (stationary)")
    a.set_title("Temporal: levels vs time"); a.set_ylabel("dB"); a.legend(fontsize=8); a.grid(alpha=.3)
    a2=ax[1]; a2.plot(g,sm(difflv,int(.6/dt)),color="#2ca02c"); a2.axhline(np.nanmedian(difflv),color="k",ls="--",lw=.7)
    a2.set_title(f"Differential level (hunter−reference): removes source's own power swings  [std {np.nanstd(difflv):.1f} dB]")
    a2.set_ylabel("Δ dB"); a2.set_xlabel("s"); a2.grid(alpha=.3)
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(os.path.join(ROOT,"amp_analysis.png"),dpi=120); print("saved amp_analysis.png")

    # stats
    print(f"raw level swing {np.nanmax(plv)-np.nanmin(plv):.1f} dB; differential level std {np.nanstd(difflv):.1f} dB")
    rr=np.corrcoef(sm(plv,5)[np.isfinite(plv)&np.isfinite(rlv)],sm(rlv,5)[np.isfinite(plv)&np.isfinite(rlv)])[0,1]
    print(f"corr(hunter level, reference level) = {rr:+.2f}")

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
