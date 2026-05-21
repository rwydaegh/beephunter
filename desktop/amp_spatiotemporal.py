"""Separate SPATIAL from TEMPORAL amplitude variation using the back-and-forth
walk + the stationary reference.

- Temporal magnitude: the stationary reference's level variation is purely
  temporal (it didn't move). And for revisited locations, the spread of hunter
  level ACROSS passes at the same spot = temporal+multipath there.
- Spatial signal: bin the track along its principal axis; the MEAN level per bin
  (averaged over passes/time) is the spatial profile with temporal averaged out.
Also re-draws the maps zoomed to the walk extent.
"""
import io, json, glob, math, os, sys, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bh_config import HOME, CONTACT, DOPSRC
DL=os.path.expanduser(r"~/Downloads"); UA={"User-Agent":f"beephunter-dev/1.0 ({CONTACT})"}
C=343.0; Z=19
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def newest(role): return sorted(glob.glob(os.path.join(DL,f"beephunter_{role}_*.json")),key=os.path.getmtime,reverse=True)[0]
def load(f):
    d=json.load(open(f,encoding="utf-8")); r=d["rows"]; g=lambda k:np.array([x.get(k) for x in r],float)
    return {"t":g("tw")/1000,"f":g("f"),"lvl":g("lvl"),"lat":g("lat"),"lon":g("lon"),"spd":g("spd")}
def sm(x,k):
    o=np.copy(x)
    for i in range(len(x)): s=x[max(0,i-k):min(len(x),i+k+1)]; s=s[np.isfinite(s)]; o[i]=np.mean(s) if len(s) else np.nan
    return o
def deg2num(lat,lon,z): n=2**z; return (lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n
def tile(z,x,y): return Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",headers=UA),timeout=20).read())).convert("RGB")
def basemap(lat,lon,latm,lonm):
    lats=list(lat)+[HOME[0],DOPSRC[0]]; lons=list(lon)+[HOME[1],DOPSRC[1]]
    la0,la1=min(lats)-latm,max(lats)+latm; lo0,lo1=min(lons)-lonm,max(lons)+lonm
    x0,y0=[int(v) for v in deg2num(la1,lo0,Z)]; x1,y1=[int(v) for v in deg2num(la0,lo1,Z)]
    W,H=(x1-x0+1)*256,(y1-y0+1)*256; cv=Image.new("RGB",(W,H))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            try: cv.paste(tile(Z,tx,ty),((tx-x0)*256,(ty-y0)*256))
            except Exception as e: print("tile",e)
    px=lambda la,lo:((deg2num(la,lo,Z)[0]-x0)*256,(deg2num(la,lo,Z)[1]-y0)*256)
    return np.asarray(cv),W,H,px,(la0,la1,lo0,lo1)
def zoom_map(lat,lon,vals,cmap,vlim,title,fname,cbar):
    img,W,H,px,bb=basemap(lat,lon,0.00010,0.00016)
    fig,ax=plt.subplots(figsize=(W/95,H/95),dpi=140); ax.imshow(img,origin="upper")
    Pp=[px(lat[i],lon[i]) for i in range(len(lat))]
    for i in range(1,len(Pp)):
        v=vals[i]; col=(.5,.5,.5,1) if not np.isfinite(v) else cmap((np.clip(v,vlim[0],vlim[1])-vlim[0])/(vlim[1]-vlim[0]))
        ax.plot([Pp[i-1][0],Pp[i][0]],[Pp[i-1][1],Pp[i][1]],"-",color=col,lw=7,solid_capstyle="round")
    for (la,lo,c,mk,lab) in [(*HOME,"#19d219","*","HOME #24"),(*DOPSRC,"lime","X","Doppler guess")]:
        mx,my=px(la,lo); ax.plot(mx,my,mk,ms=22,color=c,mec="k",mew=1.4); ax.annotate(lab,(mx,my),color="#0a4d0a",fontsize=12,fontweight="bold",xytext=(8,-12),textcoords="offset points")
    smp=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(*vlim)); smp.set_array([]); plt.colorbar(smp,ax=ax,label=cbar,shrink=.7)
    ax.set_title(title,fontsize=13,fontweight="bold"); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT,fname),dpi=140,bbox_inches="tight"); print("saved",fname)

def main():
    P=load(newest("hunter")); R=load(newest("reference"))
    dt=0.2; t0=max(P["t"][0],R["t"][0]); t1=min(P["t"][-1],R["t"][-1]); g=np.arange(0,t1-t0,dt)
    plv=np.interp(g,P["t"]-t0,P["lvl"]); rlv=np.interp(g,R["t"]-t0,R["lvl"])
    lat=np.interp(g,P["t"]-t0,P["lat"]); lon=np.interp(g,P["t"]-t0,P["lon"]); spd=np.interp(g,P["t"]-t0,np.nan_to_num(P["spd"],nan=0))
    mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat.mean()))
    X=(lon-lon.mean())*mlon; Y=(lat-lat.mean())*mlat
    pts=np.column_stack([X,Y]); _,_,V=np.linalg.svd(pts-pts.mean(0),full_matrices=False); axis=V[0]
    s=(pts-pts.mean(0))@axis   # along-street coordinate (m)

    # ---- temporal magnitude ----
    print(f"TEMPORAL (reference, stationary): std {np.nanstd(rlv):.1f} dB, 5-95%% range {np.nanpercentile(rlv,95)-np.nanpercentile(rlv,5):.1f} dB")
    print(f"hunter level: std {np.nanstd(plv):.1f} dB, full range {np.nanmax(plv)-np.nanmin(plv):.1f} dB")

    # ---- bin along-street; mean (spatial) vs within-bin std (temporal at revisits) ----
    bw=3.0; edges=np.arange(s.min(),s.max()+bw,bw); cen=0.5*(edges[:-1]+edges[1:])
    mean=np.full(len(cen),np.nan); sd=np.full(len(cen),np.nan); npass=np.zeros(len(cen))
    for j in range(len(cen)):
        m=(s>=edges[j])&(s<edges[j+1])&np.isfinite(plv)
        if m.sum()>=4: mean[j]=np.nanmean(plv[m]); sd[j]=np.nanstd(plv[m]); npass[j]=m.sum()
    within=np.nanmean(sd); between=np.nanstd(mean)
    print(f"SPATIAL between-bin std (temporal-averaged profile): {between:.1f} dB")
    print(f"TEMPORAL within-bin std (same spot, different passes): {within:.1f} dB")
    print(f"=> spatial-signal / temporal-noise ratio ≈ {between/within:.2f}")
    # source / home positions along axis
    def sof(la,lo): p=np.array([(lo-lon.mean())*mlon,(la-lat.mean())*mlat]); return (p-pts.mean(0))@axis
    s_src=sof(*DOPSRC); s_home=sof(*HOME); s_peak=cen[np.nanargmax(mean)]
    print(f"level-peak at s={s_peak:.0f}m | Doppler-guess at s={s_src:.0f}m | home at s={s_home:.0f}m")

    # ============ FIG: temporal + spatial ============
    fig,ax=plt.subplots(2,2,figsize=(14,9)); fig.suptitle("Amplitude: temporal vs spatial decomposition",fontweight="bold")
    a=ax[0,0]; a.plot(g,sm(rlv,3),color="#d62728"); a.set_title(f"Reference level vs time (stationary ⇒ pure temporal)  std={np.nanstd(rlv):.1f} dB"); a.set_xlabel("s"); a.set_ylabel("dB"); a.grid(alpha=.3)
    a=ax[0,1]; a.hist(rlv[np.isfinite(rlv)],bins=40,color="#d62728",alpha=.5,label="reference (temporal)"); a.hist(plv[np.isfinite(plv)],bins=40,color="#1f77b4",alpha=.5,label="hunter (spatial+temporal)")
    a.set_title("Level distributions"); a.set_xlabel("dB"); a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[1,0]; a.plot(g,sm(plv,3),color="#1f77b4",label="hunter"); a.plot(g,sm(rlv,3),color="#d62728",alpha=.6,label="reference")
    a.set_title("Levels vs time (uncorrelated ⇒ temporal not common)"); a.set_xlabel("s"); a.set_ylabel("dB"); a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[1,1]; a.plot(cen,mean,"o-",color="#2ca02c"); a.fill_between(cen,mean-sd,mean+sd,color="#2ca02c",alpha=.2)
    a.axvline(s_src,color="lime",ls="--",label="Doppler guess"); a.axvline(s_home,color="#19d219",ls=":",label="home")
    a.set_title(f"SPATIAL profile along street (mean±std per 3m bin)\nspatial {between:.1f}dB vs temporal {within:.1f}dB"); a.set_xlabel("along-street position (m)"); a.set_ylabel("mean level (dB)"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(os.path.join(ROOT,"amp_decompose.png"),dpi=120); print("saved amp_decompose.png")

    # ============ zoomed maps ============
    a_lo,a_hi=np.nanpercentile(plv,5),np.nanpercentile(plv,95)
    zoom_map(lat,lon,sm(plv,int(1.0/dt)),plt.cm.inferno,(a_lo,a_hi),"Band level (zoomed to walk)","map_level_zoom.png","level (dB)")

if __name__=="__main__":
    try: sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    except Exception: pass
    main()
