/* BeepHunter analyzer — phone-only Doppler localization (reference optional).
 *
 * The stationary test showed the amplitude channel is multipath-corrupted
 * (anti-correlated between devices), but the FREQUENCY is stable to ~0.2 Hz, so
 * Doppler is the robust localizer. At closest approach the radial velocity is
 * zero -> Doppler crosses its rest value -> the source is abeam (perpendicular
 * to heading). Each crossing is a line; we least-squares intersect them.
 *
 * rest f0: from stationary moments (speed<0.3) if available, else the reference
 * device's median, else the overall median of the walk.
 */
'use strict';
const $ = id => document.getElementById(id);
const C_SOUND = 343.0;
let refData=null, huntData=null, map=null, layers=[];

$('huntFile').onchange = e => load(e.target.files[0], 'hunt');
$('refFile').onchange  = e => load(e.target.files[0], 'ref');

function load(file, which){
  if(!file) return; const r=new FileReader();
  r.onload=()=>{ try{ const j=JSON.parse(r.result);
    if(which==='ref') refData=j; else huntData=j;
    $('msg').textContent=`${which} loaded (${j.rows.length} rows, role=${j.meta?.role||'?'})`;
    if(huntData) run();
  }catch(err){ $('msg').textContent='parse error: '+err.message; } };
  r.readAsText(file);
}

const series=(d,f)=> d.rows.filter(r=>r[f]!=null).map(r=>({t:r.tw, v:r[f]}));
function resample(s,t0,t1,dt){ const out=[]; let j=0; if(!s.length){ for(let t=t0;t<=t1;t+=dt) out.push(NaN); return out; }
  for(let t=t0;t<=t1;t+=dt){ while(j<s.length-1&&s[j+1].t<t) j++;
    if(j>=s.length-1){ out.push(s[s.length-1].v); continue; }
    const a=s[j],b=s[j+1],f=(t-a.t)/Math.max(1,b.t-a.t); out.push(a.v+f*(b.v-a.v)); } return out; }
const median=a=>{ const s=[...a].filter(isFinite).sort((x,y)=>x-y); return s.length?s[Math.floor(s.length/2)]:NaN; };
function smooth(a,k){ const o=[...a]; for(let i=0;i<a.length;i++){ let s=0,n=0; for(let j=-k;j<=k;j++){const m=i+j; if(m>=0&&m<a.length&&isFinite(a[m])){s+=a[m];n++;}} o[i]=n?s/n:NaN; } return o; }
function projector(lat0,lon0){ const mlat=111320,mlon=111320*Math.cos(lat0*Math.PI/180);
  return {xy:(la,lo)=>[(lo-lon0)*mlon,(la-lat0)*mlat], ll:(x,y)=>[lat0+y/mlat,lon0+x/mlon]}; }
function nearestRow(rows,tw){ let best=null,bd=1e15; for(const r of rows){ const d=Math.abs(r.tw-tw); if(d<bd){bd=d;best=r;} } return best; }
function localHeading(rows,tw){ const a=nearestRow(rows,tw-700),b=nearestRow(rows,tw+700);
  if(!a||!b||a.lat==null||b.lat==null) return null; return (Math.atan2(b.lon-a.lon,b.lat-a.lat)*180/Math.PI+360)%360; }
function localSpeed(rows,tw,P){ const a=nearestRow(rows,tw-700),b=nearestRow(rows,tw+700);
  if(!a||!b||a.lat==null||b.lat==null) return 0; const pa=P.xy(a.lat,a.lon),pb=P.xy(b.lat,b.lon);
  return Math.hypot(pb[0]-pa[0],pb[1]-pa[1])/Math.max(0.1,(b.tw-a.tw)/1000); }

function run(){
  if(!series(huntData,'f').length){ $('msg').innerHTML='<span style="color:#f87272">hunter file has no frequency data — re-record with the latest app.</span>'; return; }
  const dt=200;
  const t0=huntData.rows[0].tw, t1=huntData.rows.at(-1).tw;
  const huntF=resample(series(huntData,'f'),t0,t1,dt);
  const huntS=resample(series(huntData,'spd'),t0,t1,dt);

  // rest f0
  let f0, src;
  const still=huntData.rows.filter(r=>r.f!=null && (r.spd==null||r.spd<0.3)).map(r=>r.f);
  if(refData && series(refData,'f').length){ f0=median(series(refData,'f').map(x=>x.v)); src='reference median'; }
  else if(still.length>20){ f0=median(still); src='stationary moments'; }
  else { f0=median(huntF); src='walk median'; }
  $('f0').textContent=f0.toFixed(2);

  const doppler=huntF.map(v=> isFinite(v)? v-f0 : NaN);
  const dop=smooth(doppler,Math.round(1500/dt));   // ~1.5s smoothing
  $('swing').textContent=(Math.max(...dop.filter(isFinite))-Math.min(...dop.filter(isFinite))).toFixed(1);
  $('lag').textContent = refData? '(ref used for f₀)':'n/a (phone-only)';

  drawTrace($('lvlCv'), huntS, '#9b8cff', 0);
  drawDopTrace($('dopCv'), dop);

  // map
  const pts=huntData.rows.filter(r=>r.lat!=null);
  if(map){ layers.forEach(l=>map.removeLayer(l)); layers=[]; }
  if(pts.length<2){ $('msg').textContent='hunter has no GPS — Doppler map needs phone location.'; return; }
  const lat0=pts.reduce((s,p)=>s+p.lat,0)/pts.length, lon0=pts.reduce((s,p)=>s+p.lon,0)/pts.length;
  const P=projector(lat0,lon0);
  if(!map){ map=L.map('map',{maxZoom:21}).setView([lat0,lon0],19);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxNativeZoom:19,maxZoom:21,keepBuffer:8,attribution:'© OpenStreetMap'}).addTo(map);
    setTimeout(()=>map.invalidateSize(),300); }
  const dAt=tw=>{ const i=Math.round((tw-t0)/dt); return dop[Math.max(0,Math.min(dop.length-1,i))]; };
  const dColor=v=>{ if(!isFinite(v)) return '#888'; const f=Math.max(-1,Math.min(1,v/8));
    return f>0?`rgb(${200+Math.round(55*f)},${Math.round(80*(1-f))},${Math.round(60*(1-f))})`
              :`rgb(${Math.round(60*(1+f))},${Math.round(120*(1+f))},${200+Math.round(55*(-f))})`; };
  for(let i=1;i<pts.length;i++){ const pl=L.polyline([[pts[i-1].lat,pts[i-1].lon],[pts[i].lat,pts[i].lon]],
    {color:dColor(dAt(pts[i].tw)),weight:7,opacity:.9}).addTo(map); layers.push(pl); }

  // abeam crossings -> perpendicular lines -> LS intersection
  const lines=[]; let crossings=0;
  for(let i=2;i<dop.length-2;i++){
    if(isFinite(dop[i-1])&&isFinite(dop[i])&&(dop[i-1]>0)!==(dop[i]>0)){
      // require a real swing around the crossing (not noise wiggling near zero)
      const w0=Math.max(0,i-Math.round(4000/dt)), w1=Math.min(dop.length-1,i+Math.round(4000/dt));
      const seg=dop.slice(w0,w1).filter(isFinite);
      if(Math.max(...seg)<2 || Math.min(...seg)>-2) continue;
      const tw=t0+i*dt, row=nearestRow(huntData.rows,tw);
      if(!row||row.lat==null) continue;
      const spd=(row.spd!=null&&row.spd>0)?row.spd:localSpeed(huntData.rows,tw,P);
      if(spd<0.4) continue;
      let hdg=row.hdg!=null?row.hdg:localHeading(huntData.rows,tw); if(hdg==null) continue;
      const a=hdg*Math.PI/180, n=[Math.sin(a),Math.cos(a)], p=P.xy(row.lat,row.lon);
      lines.push({p,n}); crossings++;
      const m=L.circleMarker([row.lat,row.lon],{radius:8,color:'#fff',weight:2,fillColor:'#ff3',fillOpacity:1}).addTo(map).bindTooltip('abeam'); layers.push(m);
      const perp=[-n[1],n[0]], e1=P.ll(p[0]+perp[0]*45,p[1]+perp[1]*45), e2=P.ll(p[0]-perp[0]*45,p[1]-perp[1]*45);
      layers.push(L.polyline([e1,e2],{color:'#ff3',weight:1.5,dashArray:'4 4',opacity:.85}).addTo(map));
    }
  }
  $('ncross').textContent=crossings;

  if(lines.length>=2){
    let A=[[0,0],[0,0]], b=[0,0];
    for(const {p,n} of lines){ const c=n[0]*p[0]+n[1]*p[1];
      A[0][0]+=n[0]*n[0];A[0][1]+=n[0]*n[1];A[1][0]+=n[1]*n[0];A[1][1]+=n[1]*n[1]; b[0]+=n[0]*c;b[1]+=n[1]*c; }
    const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
    if(Math.abs(det)>1e-6){ const qx=(b[0]*A[1][1]-A[0][1]*b[1])/det, qy=(A[0][0]*b[1]-b[0]*A[1][0])/det;
      const [slat,slon]=P.ll(qx,qy);
      layers.push(L.circleMarker([slat,slon],{radius:11,color:'#000',weight:3,fillColor:'#0f0',fillOpacity:1}).addTo(map).bindTooltip('★ Doppler source estimate',{permanent:true}));
      const lg=L.control({position:'topright'}); lg.onAdd=()=>{const d=L.DomUtil.create('div','lg');
        d.innerHTML=`<b>★ source estimate</b><br>${slat.toFixed(6)}, ${slon.toFixed(6)}<br>${lines.length} abeam lines · f₀=${f0.toFixed(2)} (${src})`; return d;};
      lg.addTo(map); layers.push(lg);
      $('explain').textContent=`Source ≈ ${slat.toFixed(6)}, ${slon.toFixed(6)} from ${lines.length} Doppler abeam crossings (f₀=${f0.toFixed(2)} Hz via ${src}). Track colored red=approaching, blue=receding; white dots = abeam (perpendicular to your path). Walk more straight transects from different angles to tighten it.`;
    }
  } else {
    $('explain').textContent=`Only ${crossings} usable abeam crossing(s) (need ≥2 from different directions). Walk straight, brisk transects past the area; the Doppler must clearly swing + → − (red→blue). f₀=${f0.toFixed(2)} Hz via ${src}.`;
  }
}

// ---- canvas helpers ----
function fit(cv){ const r=cv.getBoundingClientRect(); cv.width=r.width*devicePixelRatio; cv.height=r.height*devicePixelRatio; }
function drawTrace(cv,arr,col,mid){ fit(cv); const c=cv.getContext('2d'),w=cv.width,h=cv.height; c.clearRect(0,0,w,h);
  const v=arr.filter(isFinite); if(v.length<2) return; let mn=Math.min(...v),mx=Math.max(...v); if(mx-mn<1e-6){mx+=1;mn-=1;}
  const Y=x=>h-(x-mn)/(mx-mn)*h*0.9-h*0.05;
  c.strokeStyle=col;c.lineWidth=2*devicePixelRatio;c.beginPath();
  arr.forEach((x,i)=>{ if(!isFinite(x))return; const px=i/(arr.length-1)*w,py=Y(x); i?c.lineTo(px,py):c.moveTo(px,py); }); c.stroke();
  c.fillStyle='#5b6776';c.font=`${11*devicePixelRatio}px sans-serif`; c.fillText(mx.toFixed(1),4,12*devicePixelRatio); c.fillText(mn.toFixed(1),4,h-4*devicePixelRatio); }
function drawDopTrace(cv,arr){ fit(cv); const c=cv.getContext('2d'),w=cv.width,h=cv.height; c.clearRect(0,0,w,h);
  const v=arr.filter(isFinite); if(v.length<2) return; const lim=Math.max(2,Math.max(...v.map(Math.abs)))*1.1;
  const Y=x=>h/2-x/lim*(h/2-4);
  c.strokeStyle='#39506a';c.lineWidth=1*devicePixelRatio;c.beginPath();c.moveTo(0,Y(0));c.lineTo(w,Y(0));c.stroke();
  for(let i=1;i<arr.length;i++){ if(!isFinite(arr[i])||!isFinite(arr[i-1]))continue; const x0=(i-1)/(arr.length-1)*w,x1=i/(arr.length-1)*w;
    c.strokeStyle=arr[i]>0.3?'#f87272':(arr[i]<-0.3?'#3abff8':'#9fb0c0'); c.lineWidth=2*devicePixelRatio;
    c.beginPath();c.moveTo(x0,Y(arr[i-1]));c.lineTo(x1,Y(arr[i]));c.stroke(); }
  c.fillStyle='#5b6776';c.font=`${11*devicePixelRatio}px sans-serif`; c.fillText('+'+lim.toFixed(0)+' Hz',4,12*devicePixelRatio); c.fillText('-'+lim.toFixed(0)+' Hz',4,h-4*devicePixelRatio); }
