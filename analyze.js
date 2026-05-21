/* BeepHunter analyzer (v1, experimental — expect to refine on real data).
 *
 * Two recordings:
 *   reference (stationary) = live rest-frequency + level reference
 *   hunter   (walking)     = level + fine frequency + GPS track
 *
 * Differential cancels the source's own behaviour:
 *   diffLevel(t)  = hunterLevel - refLevel      -> spatial proximity (temporal modulation removed)
 *   doppler(t)    = hunterFreq  - refFreq       -> pure Doppler (intrinsic drift removed)
 * Clocks are auto-aligned by cross-correlating the two level traces (both see
 * the same source amplitude modulation).
 *
 * Localization: at closest approach the radial velocity (Doppler) is zero, so
 * the source is abeam — perpendicular to the walking heading at that instant.
 * Each zero-crossing gives a line; we least-squares intersect them.
 */
'use strict';
const $ = id => document.getElementById(id);
const C_SOUND = 343.0;
let refData=null, huntData=null, map=null, layers=[];

$('refFile').onchange  = e => load(e.target.files[0], 'ref');
$('huntFile').onchange = e => load(e.target.files[0], 'hunt');

function load(file, which){
  if(!file) return;
  const r=new FileReader();
  r.onload=()=>{ try{
    const j=JSON.parse(r.result);
    // auto-detect role from meta if mislabeled
    const role=j.meta?.role;
    if(role==='reference') refData=j; else if(role==='hunter') huntData=j;
    else (which==='ref'?refData=j:huntData=j);
    $('msg').textContent = `${which} loaded (${j.rows.length} rows, role=${role||'?'})`;
    if(refData && huntData) run();
  }catch(err){ $('msg').textContent='parse error: '+err.message; } };
  r.readAsText(file);
}

// ---- resample a recording's field onto a time grid (ms wall-clock) ----------
function series(data, field){
  return data.rows.filter(r=>r[field]!=null).map(r=>({t:r.tw, v:r[field]}));
}
function resample(s, t0, t1, dt){
  const out=[]; let j=0;
  for(let t=t0;t<=t1;t+=dt){
    while(j<s.length-1 && s[j+1].t<t) j++;
    if(j>=s.length-1){ out.push(s[Math.min(j,s.length-1)].v); continue; }
    const a=s[j], b=s[j+1], f=(t-a.t)/Math.max(1,(b.t-a.t));
    out.push(a.v + f*(b.v-a.v));
  }
  return out;
}
// cross-correlation -> integer lag (in samples of dt) maximizing alignment
function bestLag(a, b, maxLag){
  const za=zmean(a), zb=zmean(b); let best=0, bestv=-Infinity;
  for(let L=-maxLag;L<=maxLag;L++){
    let s=0,n=0;
    for(let i=0;i<za.length;i++){ const k=i+L; if(k<0||k>=zb.length) continue; s+=za[i]*zb[k]; n++; }
    if(n>10){ s/=n; if(s>bestv){bestv=s;best=L;} }
  }
  return best;
}
const zmean = a => { const m=a.reduce((x,y)=>x+y,0)/a.length; return a.map(v=>v-m); };
const median = a => { const s=[...a].sort((x,y)=>x-y); return s[Math.floor(s.length/2)]; };

// ---- geo helpers -----------------------------------------------------------
function projector(lat0,lon0){
  const mlat=111320, mlon=111320*Math.cos(lat0*Math.PI/180);
  return { xy:(la,lo)=>[(lo-lon0)*mlon,(la-lat0)*mlat],
           ll:(x,y)=>[lat0+y/mlat, lon0+x/mlon] };
}

function run(){
  // common overlapping window
  const dt=200; // ms grid (5 Hz)
  const t0=Math.max(refData.rows[0].tw, huntData.rows[0].tw);
  const t1=Math.min(refData.rows.at(-1).tw, huntData.rows.at(-1).tw);
  if(t1-t0 < 3000){ $('msg').textContent='recordings barely overlap in time — check clocks'; }

  const refL=resample(series(refData,'lvl'),t0,t1,dt);
  const huntL=resample(series(huntData,'lvl'),t0,t1,dt);
  const refF=resample(series(refData,'f'),t0,t1,dt);
  const huntF=resample(series(huntData,'f'),t0,t1,dt);

  // auto clock sync via level cross-correlation
  const lag=bestLag(huntL,refL,Math.round(5000/dt)); // ±5 s
  const shift = arr => arr.map((_,i)=> arr[Math.max(0,Math.min(arr.length-1,i+lag))]);
  const refLs=shift(refL), refFs=shift(refF);

  const diffLevel = huntL.map((v,i)=> v-refLs[i]);
  const doppler   = huntF.map((v,i)=> v-refFs[i]);
  const f0 = median(refF);
  $('f0').textContent=f0.toFixed(2); $('lag').textContent=(lag*dt/1000).toFixed(2);
  $('swing').textContent=(Math.max(...doppler)-Math.min(...doppler)).toFixed(1);

  drawTrace($('lvlCv'), diffLevel, '#36d399', 0);
  drawTrace($('dopCv'), doppler, '#fbbd23', 0, f0/C_SOUND); // overlay ±v scale ticks

  // map hunter GPS track, colored by differential level
  const pts = huntData.rows.filter(r=>r.lat!=null);
  if(map){ layers.forEach(l=>map.removeLayer(l)); layers=[]; }
  if(pts.length<2){ $('msg').textContent='hunter has no GPS — map needs phone location'; return; }
  const lat0=pts.reduce((s,p)=>s+p.lat,0)/pts.length, lon0=pts.reduce((s,p)=>s+p.lon,0)/pts.length;
  const P=projector(lat0,lon0);
  if(!map){ map=L.map('map',{maxZoom:21}).setView([lat0,lon0],19);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      {maxNativeZoom:19,maxZoom:21,keepBuffer:8,attribution:'© OpenStreetMap'}).addTo(map);
    setTimeout(()=>map.invalidateSize(),300);
  }
  // color track by diffLevel (interpolate diffLevel onto hunter rows by time)
  const dlMin=percentile(diffLevel,5), dlMax=percentile(diffLevel,95);
  const dlAt = tw => { const i=Math.round((tw-t0)/dt); return diffLevel[Math.max(0,Math.min(diffLevel.length-1,i))]; };
  const color = v => { let f=(v-dlMin)/(dlMax-dlMin+1e-9); f=Math.max(0,Math.min(1,f));
    return `rgb(${Math.round(255*f)},${Math.round(60*(1-Math.abs(f-.5)*2))},${Math.round(255*(1-f))})`; };
  for(let i=1;i<pts.length;i++){
    const pl=L.polyline([[pts[i-1].lat,pts[i-1].lon],[pts[i].lat,pts[i].lon]],
      {color:color(dlAt(pts[i].tw)),weight:7,opacity:.9}).addTo(map); layers.push(pl);
  }
  pts.forEach(p=>{ const m=L.circleMarker([p.lat,p.lon],{radius:3,color:'#111',weight:.5,
    fillColor:color(dlAt(p.tw)),fillOpacity:.95}).addTo(map); layers.push(m); });

  // ---- Doppler abeam crossings -> perpendicular lines -> LS intersection ----
  const dop = smooth(doppler, 5);
  const lines=[]; // {p:[x,y], n:[nx,ny]}  n = velocity (heading) unit vector
  let crossings=0;
  for(let i=2;i<dop.length-2;i++){
    if((dop[i-1]>0)!==(dop[i]>0)){ // sign change = abeam
      const tw=t0+i*dt;
      const row=nearestRow(huntData.rows,tw);
      if(!row || row.lat==null) continue;
      const spd=row.spd||localSpeed(huntData.rows,tw,P);
      if(spd<0.4) continue; // must be moving to be meaningful
      let hdg=row.hdg;
      if(hdg==null) hdg=localHeading(huntData.rows,tw); // derive from track
      if(hdg==null) continue;
      const a=hdg*Math.PI/180; const n=[Math.sin(a),Math.cos(a)]; // heading unit (x=E,y=N)
      const p=P.xy(row.lat,row.lon);
      lines.push({p,n}); crossings++;
      const m=L.circleMarker([row.lat,row.lon],{radius:8,color:'#fff',weight:2,fillColor:'#ff3',fillOpacity:1})
        .addTo(map).bindTooltip('abeam crossing'); layers.push(m);
      // draw the perpendicular-to-heading line (where the source lies)
      const perp=[-n[1],n[0]];
      const e1=P.ll(p[0]+perp[0]*40,p[1]+perp[1]*40), e2=P.ll(p[0]-perp[0]*40,p[1]-perp[1]*40);
      const ln=L.polyline([e1,e2],{color:'#ff3',weight:1.5,dashArray:'4 4',opacity:.8}).addTo(map); layers.push(ln);
    }
  }
  $('ncross').textContent=crossings;

  // least-squares intersection of the abeam lines (source lies on each)
  if(lines.length>=2){
    let A=[[0,0],[0,0]], b=[0,0];
    for(const {p,n} of lines){
      const c=n[0]*p[0]+n[1]*p[1];
      A[0][0]+=n[0]*n[0]; A[0][1]+=n[0]*n[1]; A[1][0]+=n[1]*n[0]; A[1][1]+=n[1]*n[1];
      b[0]+=n[0]*c; b[1]+=n[1]*c;
    }
    const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
    if(Math.abs(det)>1e-6){
      const qx=( b[0]*A[1][1]-A[0][1]*b[1])/det, qy=(A[0][0]*b[1]-b[0]*A[1][0])/det;
      const [slat,slon]=P.ll(qx,qy);
      const star=L.circleMarker([slat,slon],{radius:11,color:'#000',weight:3,fillColor:'#0f0',fillOpacity:1})
        .addTo(map).bindTooltip('★ Doppler source estimate',{permanent:true}); layers.push(star);
      const lg=L.control({position:'topright'}); lg.onAdd=()=>{const d=L.DomUtil.create('div','lg');
        d.innerHTML=`<b>★ source estimate</b><br>${slat.toFixed(6)}, ${slon.toFixed(6)}<br>from ${lines.length} abeam lines`; return d;};
      lg.addTo(map); layers.push(lg);
      $('explain').textContent = `Source estimate from ${lines.length} Doppler abeam crossings: ${slat.toFixed(6)}, ${slon.toFixed(6)}. `+
        `Differential level coloring shows proximity with the source's own loudness swings removed. `+
        `For a tighter fix, walk more straight transects from different directions through the area.`;
    }
  } else {
    $('explain').textContent = `Only ${crossings} usable abeam crossing(s). Doppler triangulation needs ≥2 straight `+
      `transects (walk briskly straight past the area in different directions). The differential-level map coloring `+
      `(temporal modulation removed) is still meaningful — head toward the red.`;
  }
}

// ---- small utils ----
function drawTrace(cv, arr, col, mid, vscale){
  fit(cv); const c=cv.getContext('2d'), w=cv.width, h=cv.height; c.clearRect(0,0,w,h);
  if(!arr.length) return;
  let mn=Math.min(...arr), mx=Math.max(...arr); if(mx-mn<1e-6){mx+=1;mn-=1;}
  const Y=v=>h-(v-mn)/(mx-mn)*h*0.9-h*0.05;
  if(mn<mid&&mx>mid){ c.strokeStyle='#39506a'; c.beginPath(); c.moveTo(0,Y(mid)); c.lineTo(w,Y(mid)); c.stroke(); }
  c.strokeStyle=col; c.lineWidth=2*devicePixelRatio; c.beginPath();
  arr.forEach((v,i)=>{ const x=i/(arr.length-1)*w, y=Y(v); i?c.lineTo(x,y):c.moveTo(x,y); }); c.stroke();
  c.fillStyle='#5b6776'; c.font=`${11*devicePixelRatio}px sans-serif`;
  c.fillText(mx.toFixed(1),4,12*devicePixelRatio); c.fillText(mn.toFixed(1),4,h-4*devicePixelRatio);
}
function fit(cv){ const r=cv.getBoundingClientRect(); cv.width=r.width*devicePixelRatio; cv.height=r.height*devicePixelRatio; }
function smooth(a,k){ const o=[...a]; for(let i=0;i<a.length;i++){ let s=0,n=0; for(let j=-k;j<=k;j++){const m=i+j; if(m>=0&&m<a.length){s+=a[m];n++;}} o[i]=s/n; } return o; }
function percentile(a,p){ const s=[...a].sort((x,y)=>x-y); return s[Math.floor(p/100*(s.length-1))]; }
function nearestRow(rows,tw){ let best=null,bd=1e15; for(const r of rows){ const d=Math.abs(r.tw-tw); if(d<bd){bd=d;best=r;} } return best; }
function localSpeed(rows,tw,P){ const a=nearestRowAround(rows,tw,-700),b=nearestRowAround(rows,tw,700);
  if(!a||!b||a.lat==null||b.lat==null) return 0; const pa=P.xy(a.lat,a.lon),pb=P.xy(b.lat,b.lon);
  return Math.hypot(pb[0]-pa[0],pb[1]-pa[1])/Math.max(0.1,(b.tw-a.tw)/1000); }
function localHeading(rows,tw){ const a=nearestRowAround(rows,tw,-700),b=nearestRowAround(rows,tw,700);
  if(!a||!b||a.lat==null||b.lat==null) return null;
  const dE=(b.lon-a.lon), dN=(b.lat-a.lat); return (Math.atan2(dE,dN)*180/Math.PI+360)%360; }
function nearestRowAround(rows,tw,off){ return nearestRow(rows,tw+off); }
