/* BeepHunter capture — runs in the browser on phone or laptop.
 * Tracks a pure tone with sub-Hz precision via complex demodulation and a
 * matched-filter "zoom" search, then shows a Doppler (sharp/flat) needle.
 *
 * Build 4 changes (from the stationary-test analysis):
 *  - Shorter 1.0 s estimation window (we had ~40:1 Doppler SNR to spare) for
 *    sharper spatial/time resolution of the abeam crossing.
 *  - Median + confidence gating rejects multipath frequency jumps.
 *  - Logs BOTH raw and smoothed frequency.
 *  - Doppler needle is the hero, with a live scrolling plot + "ABEAM" flash.
 *  - Calibrate button auto-sets rest f0 from a few stationary seconds.
 *  - center default 3036 Hz (the measured pitch); level demoted to a hint
 *    (it's multipath-corrupted and not a reliable localizer).
 */
'use strict';
const BUILD = '5';
const $ = id => document.getElementById(id);
const C_SOUND = 343.0;
const RING_SEC = 3.0;
const FINE_SEC = 1.0;             // estimation window (shorter = sharper in space)
const BB_FS = 300;                // baseband rate after decimation (Hz)
const DOP_RANGE = 18, DOP_STEP = 0.05;
const UPDATE_MS = 120;
const MED_K = 7;                  // median window for outlier rejection
const JUMP_HZ = 4.0;             // reject estimate this far from running median (unless high conf)
const CONF_MIN = 0.15;            // matched-filter coherence to trust an estimate
const DF_HIST = 260;              // scrolling doppler points (~31 s)

let ctx, stream, src, analyser, proc, tdData=null;
let fs = 48000;
let ring, ringW = 0, ringN = 0;
let freqData;
let running = false;
let restF0 = null, autoTrack = true;
let coarseF = 3036, fineEMA = 3036;
let rawHist = [], dfHist = [], calUntil = 0, calVals = [];
let lastDfSign = 0, abeamFlashUntil = 0;

let recording = false, recRows = [], recMeta = null;
let gps = {lat:null, lon:null, acc:null, spd:null, hdg:null, t:0}, gpsWatch = null;

const scope = $('scope'), sctx = scope.getContext('2d');
const dop = $('dopPlot'), dctx = dop.getContext('2d');
function fit(c){ const r=c.getBoundingClientRect(); c.width=r.width*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio; }
addEventListener('resize', ()=>{fit(scope);fit(dop);});

// ---------------------------------------------------------------- mic setup
async function listMics(){
  try{ const devs=await navigator.mediaDevices.enumerateDevices(); const sel=$('mic'); sel.innerHTML='';
    devs.filter(d=>d.kind==='audioinput').forEach(d=>{ const o=document.createElement('option');
      o.value=d.deviceId; o.textContent=d.label||('mic '+sel.length); sel.appendChild(o); });
  }catch(e){}
}
async function start(){
  try{
    const devId=$('mic').value;
    stream=await navigator.mediaDevices.getUserMedia({audio:{
      echoCancellation:false, noiseSuppression:false, autoGainControl:false,
      channelCount:1, ...(devId?{deviceId:{exact:devId}}:{}) }});
  }catch(e){ $('status').innerHTML='<span class="err">mic denied: '+e.message+'</span>'; return; }
  await listMics();
  ctx=new (window.AudioContext||window.webkitAudioContext)(); await ctx.resume();
  fs=ctx.sampleRate; ringN=Math.ceil(RING_SEC*fs); ring=new Float32Array(ringN); ringW=0;
  src=ctx.createMediaStreamSource(stream);
  analyser=ctx.createAnalyser(); analyser.fftSize=32768; analyser.smoothingTimeConstant=0;
  freqData=new Float32Array(analyser.frequencyBinCount); src.connect(analyser);
  proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{ const x=e.inputBuffer.getChannelData(0);
    for(let i=0;i<x.length;i++){ ring[ringW]=x[i]; ringW=(ringW+1)%ringN; } };
  src.connect(proc); proc.connect(ctx.destination);
  running=true;
  $('startBtn').textContent='■ Stop mic'; $('startBtn').classList.remove('go'); $('startBtn').classList.add('stop');
  startGPS(); fit(scope); fit(dop);
}
function stop(){ running=false;
  try{ proc.disconnect(); src.disconnect(); analyser.disconnect(); stream.getTracks().forEach(t=>t.stop()); ctx.close(); }catch(e){}
  $('startBtn').textContent='▶ Start mic'; $('startBtn').classList.add('go'); $('startBtn').classList.remove('stop');
  $('status').textContent='stopped.';
  if(gpsWatch!==null){ navigator.geolocation.clearWatch(gpsWatch); gpsWatch=null; }
}

// ------------------------------------------------------------------- GPS
function startGPS(){
  if(!navigator.geolocation){ $('gps').textContent='GPS: not available'; return; }
  gpsWatch=navigator.geolocation.watchPosition(p=>{ const c=p.coords;
    gps={lat:c.latitude,lon:c.longitude,acc:c.accuracy,spd:c.speed,hdg:c.heading,t:p.timestamp};
    $('gps').innerHTML=`GPS ${c.latitude.toFixed(6)}, ${c.longitude.toFixed(6)} `+
      `<span class="hint">±${(c.accuracy||0).toFixed(0)}m · ${c.speed!=null?c.speed.toFixed(1)+' m/s':'– m/s'} · ${c.heading!=null?c.heading.toFixed(0)+'°':'–°'}</span>`;
  }, err=>{ $('gps').innerHTML='<span class="err">GPS: '+err.message+'</span>'; },
  {enableHighAccuracy:true, maximumAge:500, timeout:8000});
}

// ----------------------------------------------------------- DSP helpers
function latest(n){ n=Math.min(n,ringN); const out=new Float32Array(n);
  let idx=(ringW-n+ringN)%ringN; for(let i=0;i<n;i++){ out[i]=ring[idx]; idx=(idx+1)%ringN; } return out; }

function inputLevelDb(){ if(!tdData||tdData.length!==analyser.fftSize) tdData=new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(tdData); let s=0; for(let i=0;i<tdData.length;i++) s+=tdData[i]*tdData[i];
  return 20*Math.log10(Math.sqrt(s/tdData.length)+1e-12); }

function coarsePeak(center, half){
  analyser.getFloatFrequencyData(freqData); const binHz=fs/analyser.fftSize;
  const lo=Math.max(1,Math.floor((center-half)/binHz)), hi=Math.ceil((center+half)/binHz);
  let bi=lo,bv=-Infinity;
  for(let i=lo;i<=hi&&i<freqData.length;i++) if(isFinite(freqData[i])&&freqData[i]>bv){bv=freqData[i];bi=i;}
  if(!isFinite(bv)) return {f:NaN,db:-Infinity};
  let delta=0;
  if(bi>0&&bi<freqData.length-1){ const a=freqData[bi-1],b=freqData[bi],c=freqData[bi+1];
    if(isFinite(a)&&isFinite(b)&&isFinite(c)){ const d=(a-2*b+c); if(d!==0){ delta=0.5*(a-c)/d; if(!isFinite(delta)||Math.abs(delta)>1) delta=0; } } }
  return {f:(bi+delta)*binHz, db:bv};
}

function fineFreq(f0){
  const N=Math.min(ringN,Math.floor(FINE_SEC*fs));
  const x=latest(N), D=Math.max(1,Math.round(fs/BB_FS)), M=Math.floor(N/D);
  const re=new Float64Array(M), im=new Float64Array(M), w=2*Math.PI*f0/fs;
  let n=0;
  for(let m=0;m<M;m++){ let sr=0,si=0;
    for(let k=0;k<D;k++,n++){ const ph=w*n; sr+=x[n]*Math.cos(ph); si-=x[n]*Math.sin(ph); }
    re[m]=sr/D; im[m]=si/D; }
  const env=new Float32Array(M); let sumAmp=0,etot=0;
  for(let m=0;m<M;m++){ env[m]=Math.hypot(re[m],im[m]); sumAmp+=env[m]; etot+=env[m]*env[m]; }
  const fsbb=fs/D; let bestP=-1,bestD=0;
  for(let d=-DOP_RANGE; d<=DOP_RANGE; d+=DOP_STEP){
    const wd=2*Math.PI*d/fsbb; let cr=0,ci=0;
    for(let m=0;m<M;m++){ const cc=Math.cos(wd*m), ss=Math.sin(wd*m);
      cr+=re[m]*cc+im[m]*ss; ci+=im[m]*cc-re[m]*ss; }
    const p=cr*cr+ci*ci; if(p>bestP){bestP=p;bestD=d;} }
  const conf = sumAmp>0 ? Math.min(1, Math.sqrt(bestP)/sumAmp) : 0;  // matched-filter coherence
  return {f:f0+bestD, env, fsbb, conf, level:etot/M};
}

function cadence(env,fsbb){
  if(env.length<32) return {hz:0,conf:0};
  let mean=0; for(const v of env) mean+=v; mean/=env.length;
  const x=env.map(v=>v-mean); const lagMin=Math.floor(fsbb/6), lagMax=Math.floor(fsbb/1);
  let ac0=0; for(const v of x) ac0+=v*v; if(ac0<=0) return {hz:0,conf:0};
  let bestL=0,bestV=0;
  for(let L=lagMin;L<=lagMax&&L<x.length;L++){ let s=0; for(let i=0;i+L<x.length;i++) s+=x[i]*x[i+L]; if(s>bestV){bestV=s;bestL=L;} }
  return bestL?{hz:fsbb/bestL,conf:Math.max(0,bestV/ac0)}:{hz:0,conf:0};
}
const median = a => { const s=[...a].sort((x,y)=>x-y); return s[Math.floor(s.length/2)]; };

// ------------------------------------------------------------------ drawing
function drawScope(env){ const w=scope.width,h=scope.height; sctx.clearRect(0,0,w,h); if(!env.length) return;
  let mx=1e-9; for(const v of env) if(v>mx) mx=v;
  sctx.strokeStyle='#36d399'; sctx.lineWidth=2*devicePixelRatio; sctx.beginPath();
  for(let i=0;i<env.length;i++){ const xx=i/(env.length-1)*w, yy=h-(env[i]/mx)*h*0.9-2; i?sctx.lineTo(xx,yy):sctx.moveTo(xx,yy);} sctx.stroke(); }

function drawDoppler(){ const w=dop.width,h=dop.height; dctx.clearRect(0,0,w,h);
  // y-scale fixed to +/- DOP_RANGE; zero line in middle
  const Y=v=>h/2 - Math.max(-DOP_RANGE,Math.min(DOP_RANGE,v))/DOP_RANGE*(h/2-4);
  dctx.strokeStyle='#39506a'; dctx.lineWidth=1*devicePixelRatio; dctx.beginPath(); dctx.moveTo(0,Y(0)); dctx.lineTo(w,Y(0)); dctx.stroke();
  dctx.fillStyle='#5b6776'; dctx.font=`${10*devicePixelRatio}px sans-serif`;
  dctx.fillText('SHARP +'+DOP_RANGE,4,12*devicePixelRatio); dctx.fillText('FLAT  -'+DOP_RANGE,4,h-4*devicePixelRatio);
  if(dfHist.length<2) return;
  for(let i=1;i<dfHist.length;i++){ const x0=(i-1)/(DF_HIST-1)*w, x1=i/(DF_HIST-1)*w;
    const v=dfHist[i]; dctx.strokeStyle = v>0.3?'#f87272':(v<-0.3?'#3abff8':'#9fb0c0');
    dctx.lineWidth=2*devicePixelRatio; dctx.beginPath(); dctx.moveTo(x0,Y(dfHist[i-1])); dctx.lineTo(x1,Y(v)); dctx.stroke(); }
}

// --------------------------------------------------- HUNT meter (glanceable)
function updateHunt(df, spd){
  const a=$('huntArrow'), w=$('huntWord'), n=$('huntNum'), pan=$('huntPanel'), st=$('huntState');
  if(df==null){ a.textContent='·'; a.style.color='#5b6776'; a.style.transform='scale(1)';
    w.textContent='— set rest f₀ —'; w.style.color='var(--dim)'; n.innerHTML='+0.0 <small>Hz</small>';
    pan.style.background=''; st.textContent='stand still to zero'; return; }
  const cap=8.0, mag=Math.min(Math.abs(df),cap)/cap, sc='scale('+(0.7+0.6*mag).toFixed(2)+')';
  const moving = (spd==null) || spd>0.5;
  n.innerHTML=(df>=0?'+':'')+df.toFixed(1)+' <small>Hz</small>';
  if(!moving){ a.textContent='·'; a.style.color='#5b6776'; a.style.transform='scale(1)';
    w.textContent='move to read'; w.style.color='var(--dim)'; pan.style.background=''; st.textContent='standing still — zeroing'; return; }
  st.textContent='hunting';
  if(df>0.6){ a.textContent='▲'; a.style.color='var(--hot)'; a.style.transform=sc;
    w.textContent='WARMER · keep going'; w.style.color='var(--hot)'; pan.style.background='rgba(248,114,114,'+(0.05+0.22*mag).toFixed(2)+')'; }
  else if(df<-0.6){ a.textContent='▼'; a.style.color='var(--cold)'; a.style.transform=sc;
    w.textContent='COLDER · turn around'; w.style.color='var(--cold)'; pan.style.background='rgba(58,191,248,'+(0.05+0.22*mag).toFixed(2)+')'; }
  else { a.textContent='◆'; a.style.color='#fff'; a.style.transform='scale(0.85)';
    w.textContent='ABEAM · source beside you'; w.style.color='#fff'; pan.style.background='rgba(255,255,255,.06)'; }
}

// ------------------------------------------------------------------ UI loop
function tick(){
  if(!running) return;
  const center=+$('center').value, half=+$('halfband').value;
  const inDb=inputLevelDb();
  const cp=coarsePeak(center,half);
  if(autoTrack){ if(isFinite(cp.f)) coarseF+=0.3*(cp.f-coarseF); } else coarseF=center;
  if(!isFinite(coarseF)) coarseF=center;
  const ff=fineFreq(coarseF);
  const cad=cadence(ff.env, ff.fsbb);
  const levelDb=isFinite(ff.level)&&ff.level>0?10*Math.log10(ff.level):-120;

  // --- frequency pipeline: median + confidence gating, then EMA ---
  let accepted=fineEMA;
  if(isFinite(ff.f)&&ff.level>1e-9){
    rawHist.push(ff.f); if(rawHist.length>MED_K) rawHist.shift();
    const med=median(rawHist);
    if(ff.conf>=CONF_MIN && Math.abs(ff.f-med)<JUMP_HZ) accepted=ff.f;       // trust it
    else if(ff.conf>=CONF_MIN*2) accepted=ff.f;                              // very confident -> allow jump
    else accepted=med;                                                       // outlier -> hold median
    fineEMA+=0.4*(accepted-fineEMA);
  }
  if(!isFinite(fineEMA)) fineEMA=coarseF;

  // calibration: collect a few seconds while still, set rest f0 to median
  if(calUntil>Date.now()){ calVals.push(fineEMA);
    $('restTxt').textContent=`calibrating… ${((calUntil-Date.now())/1000).toFixed(1)}s`;
  } else if(calVals.length){ restF0=median(calVals); calVals=[];
    $('restTxt').textContent='rest f₀ = '+restF0.toFixed(2)+' Hz'; }

  // auto-zero rest f0 while essentially stationary (the source pitch is now stable),
  // so the hunt meter re-zeros every time you stop walking.
  if(calUntil<=Date.now() && gps.spd!=null && gps.spd<0.3 && ff.conf>0.3 && isFinite(fineEMA)){
    if(restF0==null){ restF0=fineEMA; $('restTxt').textContent='rest f₀ ≈ '+restF0.toFixed(2)+' Hz (auto)'; }
    else restF0 += 0.05*(fineEMA-restF0);
  }

  // --- doppler ---
  let df=null;
  if(restF0!=null){ df=fineEMA-restF0; dfHist.push(df); if(dfHist.length>DF_HIST) dfHist.shift();
    $('dopVal').textContent=(df>=0?'+':'')+df.toFixed(2)+' Hz';
    const frac=Math.max(-1,Math.min(1,df/DOP_RANGE));
    $('needle').style.left=(50+frac*48)+'%';
    $('needle').style.background=df>0.3?'#f87272':(df<-0.3?'#3abff8':'#fff');
    $('dopWord').textContent = df>0.5?'APPROACHING':(df<-0.5?'RECEDING':'— abeam —');
    // abeam flash: smoothed sign change after a real swing
    if(dfHist.length>10){ const recent=dfHist.slice(-12); const sm=recent.reduce((a,b)=>a+b,0)/recent.length;
      const sgn=sm>0.4?1:(sm<-0.4?-1:0);
      if(sgn!==0 && lastDfSign!==0 && sgn!==lastDfSign){ abeamFlashUntil=Date.now()+1400; }
      if(sgn!==0) lastDfSign=sgn; }
  }
  $('abeam').style.opacity = abeamFlashUntil>Date.now() ? '1':'0';
  updateHunt(df, gps.spd);

  // --- displays ---
  $('freqHero').innerHTML=fineEMA.toFixed(2)+' <small>Hz</small>';
  $('status').textContent=`@${fs}Hz · in ${inDb.toFixed(0)}dB · conf ${ff.conf.toFixed(2)}`+(inDb<-70?' ⚠ silent':'');
  const locked=cp.db>-70 && cad.conf>0.25 && cad.hz>=1 && cad.hz<=6;
  $('badge').textContent=locked?'● LOCKED':'○ searching'; $('badge').className=locked?'locked':'';
  $('cad').textContent=cad.hz?cad.hz.toFixed(2):'–';
  $('lvl').textContent=levelDb.toFixed(0);
  drawScope(ff.env); drawDoppler();

  if(recording) recRows.push({ tw:Date.now(),
    f: isFinite(fineEMA)?+fineEMA.toFixed(3):null,
    fraw: isFinite(ff.f)?+ff.f.toFixed(3):null,
    df: df!=null?+df.toFixed(3):null,
    conf:+ff.conf.toFixed(3),
    lvl: isFinite(levelDb)?+levelDb.toFixed(2):null,
    inlvl: isFinite(inDb)?+inDb.toFixed(2):null,
    cad:+cad.hz.toFixed(2), ccf:+cad.conf.toFixed(2),
    lat:gps.lat, lon:gps.lon, acc:gps.acc, spd:gps.spd, hdg:gps.hdg });
  if(recording) $('recInfo').textContent=`recording… ${recRows.length} rows · ${((Date.now()-recMeta.started)/1000).toFixed(0)}s`;
}
setInterval(()=>{ try{tick();}catch(e){console.error(e);} }, UPDATE_MS);

// --------------------------------------------------------------- controls
$('startBtn').onclick=()=> running?stop():start();
$('calBtn').onclick=()=>{ calVals=[]; calUntil=Date.now()+3000; };
$('autoBtn').onclick=()=>{ autoTrack=!autoTrack; $('autoBtn').classList.toggle('active',autoTrack);
  $('autoBtn').textContent=autoTrack?'auto-track':'manual'; };
$('mic').onchange=()=>{ if(running){ stop(); setTimeout(start,200);} };
$('recBtn').onclick=()=>{
  if(!recording){ recording=true; recRows=[];
    recMeta={role:$('role').value, label:$('deviceLabel').value||$('role').value, fs, restF0, build:BUILD, started:Date.now(), ua:navigator.userAgent};
    $('recBtn').textContent='■ Stop recording'; $('recBtn').classList.add('stop'); $('dlBtn').disabled=true;
  } else { recording=false; $('recBtn').textContent='● Start recording'; $('recBtn').classList.remove('stop');
    recMeta.ended=Date.now(); $('dlBtn').disabled=recRows.length===0;
    $('recInfo').textContent=`stopped · ${recRows.length} rows · tap Download`; }
};
$('dlBtn').onclick=()=>{
  const blob=new Blob([JSON.stringify({meta:recMeta,rows:recRows})],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=`beephunter_${recMeta.role}_${recMeta.label}_${new Date(recMeta.started).toISOString().replace(/[:.]/g,'-').slice(0,19)}.json`; a.click();
};

if($('ver')) $('ver').textContent='build '+BUILD;
navigator.mediaDevices?.enumerateDevices && listMics();
