/* BeepHunter capture — runs in the browser on phone or laptop.
 * Tracks a pure tone with sub-Hz precision via complex demodulation,
 * shows a Doppler (sharp/flat) needle, logs level+cadence+GPS, downloads JSON.
 *
 * Why demodulation instead of a big FFT: mixing the signal down to a narrow
 * baseband around the tone lets a cheap fine-frequency search resolve far below
 * the FFT bin width, which is what Doppler (a few Hz on ~3 kHz) demands.
 */
'use strict';
const $ = id => document.getElementById(id);
const C_SOUND = 343.0;            // speed of sound, m/s
const RING_SEC = 4.0;             // rolling capture buffer
const FINE_SEC = 2.0;             // window for fine-frequency estimation
const BB_FS = 300;                // baseband rate after decimation (Hz)
const DOP_RANGE = 14;             // Hz, fine search half-range around tone
const DOP_STEP = 0.05;            // Hz, fine search resolution
const UPDATE_MS = 130;            // DSP/UI cadence

let ctx, stream, src, analyser, proc;
let fs = 48000;
let ring, ringW = 0, ringN = 0;   // Float32 ring buffer
let freqData;                     // analyser dB spectrum
let running = false;
let restF0 = null;
let autoTrack = true;
let coarseF = 3050, fineF = 3050, fineEMA = 3050;
let lastEnv = new Float32Array(0);

// recording
let recording = false, recRows = [], recMeta = null;
// gps
let gps = {lat:null, lon:null, acc:null, spd:null, hdg:null, t:0}, gpsWatch = null;

const scope = $('scope'), sctx = scope.getContext('2d');
function fitCanvas(c){ const r=c.getBoundingClientRect(); c.width=r.width*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio; }
addEventListener('resize', ()=>fitCanvas(scope));

// ---------------------------------------------------------------- mic setup
async function listMics(){
  try{
    const devs = await navigator.mediaDevices.enumerateDevices();
    const sel = $('mic'); sel.innerHTML = '';
    devs.filter(d=>d.kind==='audioinput').forEach(d=>{
      const o=document.createElement('option'); o.value=d.deviceId;
      o.textContent=d.label||('mic '+sel.length); sel.appendChild(o);
    });
  }catch(e){}
}

async function start(){
  try{
    const devId = $('mic').value;
    const constraints = { audio: {
      echoCancellation:false, noiseSuppression:false, autoGainControl:false,
      channelCount:1, ...(devId && devId!=='' ? {deviceId:{exact:devId}} : {})
    }};
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  }catch(e){ $('status').innerHTML = '<span class="err">mic denied: '+e.message+'</span>'; return; }
  await listMics();

  ctx = new (window.AudioContext||window.webkitAudioContext)();
  await ctx.resume();
  fs = ctx.sampleRate;
  ringN = Math.ceil(RING_SEC*fs); ring = new Float32Array(ringN); ringW = 0;

  src = ctx.createMediaStreamSource(stream);
  analyser = ctx.createAnalyser(); analyser.fftSize = 32768; analyser.smoothingTimeConstant = 0;
  freqData = new Float32Array(analyser.frequencyBinCount);
  src.connect(analyser);

  // ScriptProcessor: contiguous input frames -> ring buffer (output not used)
  proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = e => {
    const x = e.inputBuffer.getChannelData(0);
    for(let i=0;i<x.length;i++){ ring[ringW]=x[i]; ringW=(ringW+1)%ringN; }
  };
  src.connect(proc); proc.connect(ctx.destination);

  running = true;
  $('startBtn').textContent = '■ Stop mic'; $('startBtn').classList.remove('go'); $('startBtn').classList.add('stop');
  $('status').textContent = 'capturing @ '+fs+' Hz (AGC/NS off)';
  startGPS();
  fitCanvas(scope);
}

function stop(){
  running=false;
  try{ proc.disconnect(); src.disconnect(); analyser.disconnect(); stream.getTracks().forEach(t=>t.stop()); ctx.close(); }catch(e){}
  $('startBtn').textContent='▶ Start mic'; $('startBtn').classList.add('go'); $('startBtn').classList.remove('stop');
  $('status').textContent='stopped.';
  if(gpsWatch!==null){ navigator.geolocation.clearWatch(gpsWatch); gpsWatch=null; }
}

// ------------------------------------------------------------------- GPS
function startGPS(){
  if(!navigator.geolocation){ $('gps').textContent='GPS: not available'; return; }
  gpsWatch = navigator.geolocation.watchPosition(p=>{
    const c=p.coords;
    gps={lat:c.latitude, lon:c.longitude, acc:c.accuracy, spd:c.speed, hdg:c.heading, t:p.timestamp};
    $('gps').innerHTML = `GPS: ${c.latitude.toFixed(6)}, ${c.longitude.toFixed(6)} `+
      `<span class="hint">±${(c.accuracy||0).toFixed(0)}m · `+
      `${c.speed!=null?c.speed.toFixed(1)+' m/s':'– m/s'} · `+
      `${c.heading!=null?c.heading.toFixed(0)+'°':'–°'}</span>`;
  }, err=>{ $('gps').innerHTML='<span class="err">GPS: '+err.message+'</span>'; },
  {enableHighAccuracy:true, maximumAge:500, timeout:8000});
}

// ------------------------------------------------------- ring buffer read
function latest(n){
  n=Math.min(n,ringN); const out=new Float32Array(n);
  let idx=(ringW-n+ringN)%ringN;
  for(let i=0;i<n;i++){ out[i]=ring[idx]; idx=(idx+1)%ringN; }
  return out;
}

// ----------------------------------------------------- coarse peak (analyser)
function coarsePeak(center, half){
  analyser.getFloatFrequencyData(freqData);
  const binHz = fs/analyser.fftSize;
  const lo=Math.max(1,Math.floor((center-half)/binHz)), hi=Math.ceil((center+half)/binHz);
  let bi=lo, bv=-Infinity;
  for(let i=lo;i<=hi && i<freqData.length;i++) if(freqData[i]>bv){bv=freqData[i];bi=i;}
  // parabolic interpolation around the peak bin
  let delta=0;
  if(bi>0 && bi<freqData.length-1){
    const a=freqData[bi-1],b=freqData[bi],c=freqData[bi+1],d=(a-2*b+c);
    if(d!==0) delta=0.5*(a-c)/d;
  }
  return {f:(bi+delta)*binHz, db:bv};
}

// --------------------------------------- fine frequency via demod + zoom search
function fineFreq(f0){
  const N=Math.min(ringN, Math.floor(FINE_SEC*fs));
  const x=latest(N);
  const D=Math.max(1, Math.round(fs/BB_FS));
  const M=Math.floor(N/D);
  const re=new Float64Array(M), im=new Float64Array(M);
  const w=2*Math.PI*f0/fs;
  // demodulate to baseband and boxcar-decimate (low-pass)
  let n=0;
  for(let m=0;m<M;m++){
    let sr=0,si=0;
    for(let k=0;k<D;k++,n++){ const ph=w*n; sr+=x[n]*Math.cos(ph); si-=x[n]*Math.sin(ph); }
    re[m]=sr/D; im[m]=si/D;
  }
  // envelope (for scope/level/cadence)
  const env=new Float32Array(M);
  for(let m=0;m<M;m++) env[m]=Math.hypot(re[m],im[m]);
  lastEnv=env;
  // matched-filter zoom search for the residual offset (Doppler)
  const fsbb=fs/D; let bestP=-1,bestD=0;
  for(let d=-DOP_RANGE; d<=DOP_RANGE; d+=DOP_STEP){
    const wd=2*Math.PI*d/fsbb; let cr=0,ci=0;
    for(let m=0;m<M;m++){
      const cc=Math.cos(wd*m), ss=Math.sin(wd*m);
      cr += re[m]*cc + im[m]*ss;   // z * exp(-j wd m)
      ci += im[m]*cc - re[m]*ss;
    }
    const p=cr*cr+ci*ci; if(p>bestP){bestP=p;bestD=d;}
  }
  // confidence: peak power vs total energy
  let etot=0; for(let m=0;m<M;m++) etot+=re[m]*re[m]+im[m]*im[m];
  const conf=etot>0 ? Math.min(1, bestP/(etot*M)) : 0;
  return {f:f0+bestD, env, fsbb, conf, level:etot/M};
}

function cadence(env, fsbb){
  if(env.length<32) return {hz:0,conf:0};
  let mean=0; for(const v of env) mean+=v; mean/=env.length;
  const x=env.map(v=>v-mean);
  const lagMin=Math.floor(fsbb/6), lagMax=Math.floor(fsbb/1);
  let ac0=0; for(const v of x) ac0+=v*v; if(ac0<=0) return {hz:0,conf:0};
  let bestL=0,bestV=0;
  for(let L=lagMin; L<=lagMax && L<x.length; L++){
    let s=0; for(let i=0;i+L<x.length;i++) s+=x[i]*x[i+L];
    if(s>bestV){bestV=s;bestL=L;}
  }
  return bestL? {hz:fsbb/bestL, conf:Math.max(0,bestV/ac0)} : {hz:0,conf:0};
}

// ------------------------------------------------------------------ UI loop
function drawScope(env){
  const w=scope.width,h=scope.height; sctx.clearRect(0,0,w,h);
  if(!env.length) return;
  let mx=1e-9; for(const v of env) if(v>mx) mx=v;
  sctx.strokeStyle='#36d399'; sctx.lineWidth=2*devicePixelRatio; sctx.beginPath();
  for(let i=0;i<env.length;i++){ const xx=i/(env.length-1)*w, yy=h-(env[i]/mx)*h*0.92-2; i?sctx.lineTo(xx,yy):sctx.moveTo(xx,yy); }
  sctx.stroke();
}

function tick(){
  if(!running) return;
  const center=+$('center').value, half=+$('halfband').value;
  const cp=coarsePeak(center,half);
  if(autoTrack) coarseF = coarseF + 0.3*(cp.f-coarseF); // gentle track
  else coarseF = center;
  const ff=fineFreq(coarseF);
  fineF=ff.f; fineEMA = fineEMA + 0.35*(fineF-fineEMA);
  const cad=cadence(ff.env, ff.fsbb);
  const levelDb = 10*Math.log10(ff.level+1e-12);

  // display
  $('freqHero').innerHTML = fineEMA.toFixed(2)+' <small>Hz</small>';
  const locked = cp.db>-70 && cad.conf>0.25 && cad.hz>=1 && cad.hz<=6;
  $('badge').textContent = locked?'● LOCKED':'○ searching'; $('badge').className = locked?'locked':'';
  $('cad').textContent = cad.hz? cad.hz.toFixed(2):'–';
  $('lvl').textContent = levelDb.toFixed(1);
  $('lvlFill').style.width = Math.max(0,Math.min(100,(levelDb+80)/80*100))+'%';
  drawScope(ff.env);

  // doppler needle (relative to rest f0)
  if(restF0!=null){
    const df=fineEMA-restF0;
    $('dopVal').textContent=(df>=0?'+':'')+df.toFixed(2)+' Hz';
    const frac=Math.max(-1,Math.min(1, df/DOP_RANGE));
    $('needle').style.left=(50+frac*48)+'%';
    $('needle').style.background = df>0.3?'#f87272':(df<-0.3?'#3abff8':'#fff');
  }

  if(recording) recRows.push({
    tw:Date.now(), f:+fineEMA.toFixed(3), df:restF0!=null?+(fineEMA-restF0).toFixed(3):null,
    lvl:+levelDb.toFixed(2), cad:+cad.hz.toFixed(2), conf:+cad.conf.toFixed(2),
    lat:gps.lat, lon:gps.lon, acc:gps.acc, spd:gps.spd, hdg:gps.hdg
  });
  if(recording) $('recInfo').textContent = `recording… ${recRows.length} rows · ${(recRows.length>0?((Date.now()-recMeta.started)/1000):0).toFixed(0)}s`;
}
setInterval(()=>{ try{tick();}catch(e){console.error(e);} }, UPDATE_MS);

// --------------------------------------------------------------- controls
$('startBtn').onclick = ()=> running? stop() : start();
$('setRest').onclick = ()=>{ restF0=fineEMA; $('restTxt').textContent='rest f₀ = '+restF0.toFixed(2)+' Hz'; };
$('autoBtn').onclick = ()=>{ autoTrack=!autoTrack; $('autoBtn').classList.toggle('active',autoTrack);
  $('autoBtn').textContent = autoTrack?'auto-track':'manual'; };
$('mic').onchange = ()=>{ if(running){ stop(); setTimeout(start,200); } };

$('recBtn').onclick = ()=>{
  if(!recording){
    recording=true; recRows=[];
    recMeta={role:$('role').value, label:$('deviceLabel').value||$('role').value,
      fs, restF0, started:Date.now(), ua:navigator.userAgent};
    $('recBtn').textContent='■ Stop recording'; $('recBtn').classList.add('stop'); $('dlBtn').disabled=true;
  } else {
    recording=false; $('recBtn').textContent='● Start recording'; $('recBtn').classList.remove('stop');
    recMeta.ended=Date.now(); $('dlBtn').disabled = recRows.length===0;
    $('recInfo').textContent = `stopped · ${recRows.length} rows captured · tap Download`;
  }
};
$('dlBtn').onclick = ()=>{
  const blob=new Blob([JSON.stringify({meta:recMeta, rows:recRows})], {type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  const stamp=new Date(recMeta.started).toISOString().replace(/[:.]/g,'-').slice(0,19);
  a.download=`beephunter_${recMeta.role}_${recMeta.label}_${stamp}.json`; a.click();
};

// try to pre-list mics (labels appear only after permission)
navigator.mediaDevices?.enumerateDevices && listMics();
