// ── LOOP IT ───────────────────────────────────────────────────────────────────

const FINGER_TIPS = [8,12,16,20];
const ALL_TIPS    = [4,8,12,16,20];
const THUMB_TIP   = 4;
const FINGER_COLS = ['#ff3aaa','#ffaa00','#00cfff','#00ffaa'];
const NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const MAJOR       = [0,2,4,5,7,9,11];
const MINOR       = [0,2,3,5,7,8,10];
const MAJOR_PENTA = [0,2,4,7,9];
const MINOR_PENTA = [0,3,5,7,10];
const JOINT_HUES  = [0,30,60,90,120,150,180,210,240,270,300,330,20,50,80,110,140,170,200,230,260];
const MAJOR_P     = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const MINOR_P     = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];

// ── 1€ FILTER ─────────────────────────────────────────────────────────────────
const OE_HZ=60, OE_MINCF=1.0, OE_BETA=0.4, OE_DC=1.0;
function oeA(cf){ const te=1/OE_HZ, tau=1/(6.2832*cf); return 1/(1+tau/te); }
let smoothLms=[[],[]];
let _oeX=[[],[]], _oeY=[[],[]], _oeDX=[[],[]], _oeDY=[[],[]];
function initOE(h,raw){
  smoothLms[h]=raw.map(lm=>({x:(1-lm.x)*width,y:lm.y*height}));
  _oeX[h]=raw.map(lm=>(1-lm.x)*width);
  _oeY[h]=raw.map(lm=>lm.y*height);
  _oeDX[h]=new Array(21).fill(0);
  _oeDY[h]=new Array(21).fill(0);
}
function stepOE(h,raw){
  const da=oeA(OE_DC);
  for(let k=0;k<21;k++){
    const tx=(1-raw[k].x)*width, ty=raw[k].y*height;
    const dxr=(tx-_oeX[h][k])*OE_HZ, dyr=(ty-_oeY[h][k])*OE_HZ;
    _oeDX[h][k]+=da*(dxr-_oeDX[h][k]);
    _oeDY[h][k]+=da*(dyr-_oeDY[h][k]);
    const ax=oeA(OE_MINCF+OE_BETA*Math.abs(_oeDX[h][k]));
    const ay=oeA(OE_MINCF+OE_BETA*Math.abs(_oeDY[h][k]));
    _oeX[h][k]+=ax*(tx-_oeX[h][k]);
    _oeY[h][k]+=ay*(ty-_oeY[h][k]);
    smoothLms[h][k]={x:_oeX[h][k],y:_oeY[h][k]};
  }
}

// ── HAND STATE ────────────────────────────────────────────────────────────────
let handLandmarks=[], handLastSeen=[0,0];
let prevPalms=[null,null], velocity=[0,0], handSize=[0,0];
let jVel=[new Array(21).fill(0),new Array(21).fill(0)];
let prevJ=[Array.from({length:21},()=>({x:0,y:0})),
           Array.from({length:21},()=>({x:0,y:0}))];
const PINCH_T=0.22;
let pState=[[false,false,false,false],[false,false,false,false]];
let pCool=[[0,0,0,0],[0,0,0,0]];
let pDist=[[1,1,1,1],[1,1,1,1]];

// ── VISUAL STATE ──────────────────────────────────────────────────────────────
// Fixed-size ring buffers — NEVER grow, NEVER leak
const CAP={r:30,g:25,p:60,s:10,b:20};
let ripples=[], glows=[], particles=[], shocks=[], bursts=[];

// ── AUDIO STATE ───────────────────────────────────────────────────────────────
let actx=null, analyser=null, keyAnalyser=null;
let dataArray=null, keyDataArray=null;
let reverbBuf=null;       // created ONCE, reused forever
let vocalEQ=null, vocalOn=false;
let uploadedAudio=null, sourceNode=null;
let sBands=new Array(8).fill(0);
let beatP=0, bassP=0, midP=0, trebP=0, chromaS=0, flashA=0;
let bpm=120, lastBeat=0, beatHist=[];
let detRoot=9, detScale='minor_penta';
let chromaAcc=new Array(12).fill(0), chromaN=0;
let scaleNotes=[], masterVol=0.8, tempo=1.0;
let keyDetectTimer=0;

// ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
function clearHand(h){
  handLandmarks[h]=null;
  smoothLms[h]=[];
  _oeX[h]=[]; _oeY[h]=[]; _oeDX[h]=[]; _oeDY[h]=[];
  prevPalms[h]=null; velocity[h]=0;
  pState[h]=[false,false,false,false];
  pDist[h]=[1,1,1,1];
  jVel[h]=new Array(21).fill(0);
  prevJ[h]=Array.from({length:21},()=>({x:0,y:0}));
}

function initMediaPipe(){
  const vid=document.getElementById('camFeed');
  const mp=new Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`});
  mp.setOptions({
    maxNumHands:2,
    modelComplexity:0,
    minDetectionConfidence:0.65,
    minTrackingConfidence:0.65
  });
  mp.onResults(r=>{
    const raw=r.multiHandLandmarks||[];
    const ness=r.multiHandedness||[];
    const sorted=[null,null];
    for(let i=0;i<raw.length;i++){
      const slot=(ness[i]?.label==='Right')?0:1;
      sorted[slot]=raw[i];
    }
    // use performance.now() consistently — never mix with millis()
    const now=performance.now();
    for(let h=0;h<2;h++){
      if(sorted[h]){
        handLandmarks[h]=sorted[h];
        handLastSeen[h]=now;
        if(!smoothLms[h]||smoothLms[h].length!==21) initOE(h,sorted[h]);
        else stepOE(h,sorted[h]);
      } else {
        clearHand(h);
      }
    }
  });
  // no async/await on onFrame — fire and forget, never queue
  const cam=new Camera(vid,{
    onFrame:()=>{ mp.send({image:vid}); },
    width:320, height:240
  });
  cam.start();
}

// ── SETUP ─────────────────────────────────────────────────────────────────────
function setup(){
  createCanvas(windowWidth,windowHeight);
  frameRate(60);
  textFont('monospace');
  smoothLms=[[],[]];
  initMediaPipe();
  buildUI();
  rebuildNotes();
}

// ── AUDIO ─────────────────────────────────────────────────────────────────────
function initAudio(){
  if(actx) return;
  actx=new(window.AudioContext||window.webkitAudioContext)();
  analyser=actx.createAnalyser();
  analyser.fftSize=2048;
  analyser.smoothingTimeConstant=0.82;
  dataArray=new Uint8Array(analyser.frequencyBinCount);
  keyAnalyser=actx.createAnalyser();
  keyAnalyser.fftSize=8192;
  keyAnalyser.smoothingTimeConstant=0.9;
  keyDataArray=new Uint8Array(keyAnalyser.frequencyBinCount);

  // build reverb IR once, reuse on every note — never allocate again
  const irLen=actx.sampleRate*1.2;
  reverbBuf=actx.createBuffer(2,irLen,actx.sampleRate);
  for(let ch=0;ch<2;ch++){
    const d=reverbBuf.getChannelData(ch);
    for(let i=0;i<irLen;i++) d[i]=(Math.random()*2-1)*Math.pow(1-i/irLen,3);
  }
}

function ensureAudio(){
  initAudio();
  if(actx.state==='suspended') actx.resume().catch(()=>{});
}

function loadTrack(f){
  if(!f) return;
  if(uploadedAudio){
    uploadedAudio.pause();
    if(sourceNode){ sourceNode.disconnect(); sourceNode=null; }
  }
  ensureAudio();
  chromaAcc=new Array(12).fill(0); chromaN=0;

  uploadedAudio=new Audio(URL.createObjectURL(f));
  uploadedAudio.loop=true;
  uploadedAudio.volume=masterVol;
  uploadedAudio.crossOrigin='anonymous';
  uploadedAudio.playbackRate=tempo;
  uploadedAudio.addEventListener('canplaythrough',()=>{
    uploadedAudio.play().catch(()=>{});
  },{once:true});

  sourceNode=actx.createMediaElementSource(uploadedAudio);
  const eq1=actx.createBiquadFilter(); eq1.type='peaking'; eq1.frequency.value=1200; eq1.Q.value=1.5; eq1.gain.value=0;
  const eq2=actx.createBiquadFilter(); eq2.type='peaking'; eq2.frequency.value=2000; eq2.Q.value=1.5; eq2.gain.value=0;
  const eq3=actx.createBiquadFilter(); eq3.type='peaking'; eq3.frequency.value=3000; eq3.Q.value=1.2; eq3.gain.value=0;
  const eq4=actx.createBiquadFilter(); eq4.type='peaking'; eq4.frequency.value=4000; eq4.Q.value=1.0; eq4.gain.value=0;
  vocalEQ={eq1,eq2,eq3,eq4};
  sourceNode.connect(eq1); eq1.connect(eq2); eq2.connect(eq3); eq3.connect(eq4);
  eq4.connect(analyser); analyser.connect(keyAnalyser); keyAnalyser.connect(actx.destination);
  applyVocalState();

  const dz=document.getElementById('dropZone');
  dz.childNodes[0].textContent=f.name.length>26?f.name.substring(0,26)+'…':f.name;
  dz.style.borderColor='rgba(52,211,153,0.35)';
  dz.style.color='rgba(52,211,153,0.85)';
}

function applyVocalState(){
  if(!vocalEQ||!actx) return;
  const {eq1,eq2,eq3,eq4}=vocalEQ;
  const t=actx.currentTime;
  eq1.gain.setTargetAtTime(vocalOn?-26:0,t,0.05);
  eq2.gain.setTargetAtTime(vocalOn?-28:0,t,0.05);
  eq3.gain.setTargetAtTime(vocalOn?-24:0,t,0.05);
  eq4.gain.setTargetAtTime(vocalOn?-18:0,t,0.05);
}

// ── AUDIO ANALYSIS ────────────────────────────────────────────────────────────
function analyzeAudio(){
  if(!analyser||!dataArray) return;
  analyser.getByteFrequencyData(dataArray);
  const len=dataArray.length;
  const sp=[0,0.03,0.08,0.15,0.25,0.4,0.6,0.8,1.0];
  for(let i=0;i<8;i++){
    const lo=floor(sp[i]*len), hi=floor(sp[i+1]*len);
    let s=0; for(let j=lo;j<hi;j++) s+=dataArray[j];
    sBands[i]=lerp(sBands[i],s/((hi-lo)*255),0.25);
  }
  bassP  =lerp(bassP,  (sBands[0]+sBands[1])*.5,.3);
  midP   =lerp(midP,   (sBands[3]+sBands[4])*.5,.3);
  trebP  =lerp(trebP,  (sBands[6]+sBands[7])*.5,.3);
  beatP  =lerp(beatP,  bassP,.4);
  chromaS=lerp(chromaS,(sBands[2]+sBands[3])*13,.2);
  flashA =lerp(flashA, bassP>.65?map(bassP,.65,1,0,26):0,.35);
  if(bassP>.5&&millis()-lastBeat>200){
    const now=millis();
    if(lastBeat>0){
      beatHist.push(now-lastBeat);
      if(beatHist.length>8) beatHist.shift();
      bpm=constrain(60000/(beatHist.reduce((a,b)=>a+b)/beatHist.length),60,220);
    }
    lastBeat=now;
  }
  if(millis()-keyDetectTimer>600){ keyDetectTimer=millis(); detectKey(); }
  if(uploadedAudio){
    const td=document.getElementById('timeDisp');
    if(td) td.innerText=fmt(uploadedAudio.currentTime)+' / '+fmt(uploadedAudio.duration||0);
    const bd=document.getElementById('bpmDisp');
    if(bd) bd.innerText=floor(bpm)+' BPM';
  }
}

function detectKey(){
  if(!keyAnalyser||!keyDataArray) return;
  keyAnalyser.getByteFrequencyData(keyDataArray);
  const bHz=actx.sampleRate/keyAnalyser.fftSize;
  const frame=new Array(12).fill(0);
  for(let i=floor(60/bHz);i<min(floor(5000/bHz),keyDataArray.length);i++){
    if(keyDataArray[i]<8) continue;
    const midi=12*Math.log2(i*bHz/440)+69;
    const pc=((Math.round(midi)%12)+12)%12;
    frame[pc]+=(keyDataArray[i]/255)**2;
  }
  const fm=Math.max(...frame)||1;
  for(let i=0;i<12;i++) frame[i]/=fm;
  chromaN++;
  for(let i=0;i<12;i++) chromaAcc[i]=chromaAcc[i]*0.96+frame[i]*0.04;
  const ch=chromaN>20?chromaAcc:frame;
  let best=-999, bRoot=detRoot, bScaleD=detScale;
  for(let root=0;root<12;root++){
    let maj=0, min=0;
    for(let i=0;i<12;i++){
      maj+=ch[(i+root)%12]*MAJOR_P[i];
      min+=ch[(i+root)%12]*MINOR_P[i];
    }
    if(maj>best){ best=maj; bRoot=root; bScaleD='major_penta'; }
    if(min>best){ best=min; bRoot=root; bScaleD='minor_penta'; }
  }
  if(best>1.8&&(bRoot!==detRoot||bScaleD!==detScale)){
    detRoot=bRoot; detScale=bScaleD;
    rebuildNotes(); updateKeyDisp();
  }
}

function fmt(s){ if(!s||isNaN(s)) return'0:00'; return floor(s/60)+':'+(floor(s%60)<10?'0':'')+floor(s%60); }

// ── NOTES ─────────────────────────────────────────────────────────────────────
function rebuildNotes(){
  const iv=detScale==='major'?MAJOR:detScale==='minor'?MINOR:detScale==='major_penta'?MAJOR_PENTA:MINOR_PENTA;
  const rHz=27.5*Math.pow(2,detRoot/12)*4;
  scaleNotes=[0,1,2,3].map(i=>{
    const st=iv[i%iv.length];
    return{ freq_l:rHz*Math.pow(2,st/12), freq_r:rHz*Math.pow(2,(st+12)/12)*4, name:NOTE_NAMES[(detRoot+st)%12] };
  });
  for(let i=0;i<4;i++){
    const le=document.getElementById('ln'+i), re=document.getElementById('rn'+i);
    if(le) le.innerText=['Idx','Mid','Rng','Pnk'][i]+'  '+scaleNotes[i].name;
    if(re) re.innerText=['Idx','Mid','Rng','Pnk'][i]+'  '+scaleNotes[i].name;
  }
}

function playNote(fi,hand,vol){
  ensureAudio();
  if(!scaleNotes.length||!reverbBuf) return;
  const freq=hand===0?scaleNotes[fi].freq_l:scaleNotes[fi].freq_r;
  const dur=hand===0?0.5:0.65;

  const conv=actx.createConvolver();
  conv.buffer=reverbBuf; // shared buffer — no allocation
  const cG=actx.createGain(); cG.gain.value=0.16;
  const o1=actx.createOscillator(), o2=actx.createOscillator();
  const lfo=actx.createOscillator(), lG=actx.createGain();
  const mx=actx.createGain(), fl=actx.createBiquadFilter(), mG=actx.createGain();

  lfo.frequency.value=5.5; lG.gain.value=freq*0.006;
  lfo.connect(lG); lG.connect(o1.frequency); lG.connect(o2.frequency);
  o1.type='sine'; o1.frequency.value=freq;
  o2.type='sine'; o2.frequency.value=freq*1.0025;
  fl.type='lowpass'; fl.frequency.value=hand===0?800:3800; fl.Q.value=0.5;
  mx.gain.value=0.5; o1.connect(mx); o2.connect(mx);
  mx.connect(fl); fl.connect(mG);
  mG.connect(actx.destination);
  mG.connect(conv); conv.connect(cG); cG.connect(actx.destination);

  const t=actx.currentTime, v=constrain(vol*0.42*(1+bassP*0.12),0.02,0.85);
  mG.gain.setValueAtTime(0,t);
  mG.gain.linearRampToValueAtTime(v,t+0.06);
  mG.gain.setValueAtTime(v,t+dur*0.55);
  mG.gain.exponentialRampToValueAtTime(0.0001,t+dur);

  o1.start(t); o1.stop(t+dur);
  o2.start(t); o2.stop(t+dur);
  lfo.start(t); lfo.stop(t+dur);

  // guaranteed cleanup via setTimeout — more reliable than onended
  // fires after note is fully done + 200ms safety margin
  setTimeout(()=>{
    try{
      lfo.disconnect(); lG.disconnect();
      o1.disconnect(); o2.disconnect();
      mx.disconnect(); fl.disconnect(); mG.disconnect();
      conv.disconnect(); cG.disconnect();
    }catch(e){}
  }, (dur+0.2)*1000);
}

// ── UI ────────────────────────────────────────────────────────────────────────
function buildUI(){
  const link=document.createElement('link');
  link.rel='stylesheet'; link.href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap';
  document.head.appendChild(link);

  const bar=document.createElement('div');
  Object.assign(bar.style,{position:'fixed',top:'0',left:'0',width:'100%',height:'56px',background:'rgba(4,3,12,0.97)',backdropFilter:'blur(24px)',webkitBackdropFilter:'blur(24px)',borderBottom:'1px solid rgba(255,255,255,0.05)',display:'flex',alignItems:'center',zIndex:'1000',fontFamily:'Inter,monospace',userSelect:'none'});
  bar.innerHTML=`
    <div style="display:flex;align-items:center;gap:10px;padding:0 20px;border-right:1px solid rgba(255,255,255,0.06);height:100%;min-width:200px">
      <span style="color:rgba(255,255,255,0.18);font-size:9px;letter-spacing:.2em;font-weight:500;text-transform:uppercase">Track</span>
      <label id="dropZone" style="border:1px solid rgba(255,255,255,0.09);border-radius:8px;padding:6px 14px;cursor:pointer;color:rgba(255,255,255,0.35);font-size:11px;background:rgba(255,255,255,0.025);transition:all .2s;white-space:nowrap;font-family:Inter,monospace">
        Drop or click to load<input type="file" id="_upl" accept="audio/*" style="display:none">
      </label>
    </div>
    <div style="display:flex;align-items:center;gap:7px;padding:0 18px;border-right:1px solid rgba(255,255,255,0.06);height:100%">
      <button id="playBtn"  style="${tbtn('#34d399')}">▶</button>
      <button id="pauseBtn" style="${tbtn('rgba(255,255,255,0.25)')}">⏸</button>
      <button id="stopBtn"  style="${tbtn('rgba(255,255,255,0.25)')}">⏹</button>
      <button id="loopBtn"  style="${tbtn('rgba(255,255,255,0.25)')}">↻</button>
      <span id="timeDisp" style="color:rgba(255,255,255,0.22);font-size:11px;min-width:82px;font-family:Inter,monospace;letter-spacing:.04em">0:00 / 0:00</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;padding:0 18px;border-right:1px solid rgba(255,255,255,0.06);height:100%">
      <span style="color:rgba(255,255,255,0.18);font-size:9px;letter-spacing:.2em;text-transform:uppercase">Vol</span>
      <input id="volSlider" type="range" min="0" max="100" value="80" style="${sldr()}">
      <span id="volVal" style="color:#67e8f9;font-size:11px;min-width:30px;font-family:Inter,monospace">80%</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;padding:0 18px;border-right:1px solid rgba(255,255,255,0.06);height:100%">
      <span style="color:rgba(255,255,255,0.18);font-size:9px;letter-spacing:.2em;text-transform:uppercase">Tempo</span>
      <input id="tempoSlider" type="range" min="50" max="200" value="100" style="${sldr()}">
      <span id="tempoVal" style="color:#67e8f9;font-size:11px;min-width:38px;font-family:Inter,monospace">1.00×</span>
    </div>
    <div style="display:flex;align-items:center;gap:14px;padding:0 18px;border-right:1px solid rgba(255,255,255,0.06);height:100%">
      <span id="bpmDisp" style="color:rgba(255,255,255,0.2);font-size:10px;font-family:Inter,monospace">— BPM</span>
      <span id="keyDisp" style="color:#c084fc;font-size:10px;font-family:Inter,monospace;min-width:110px">KEY  —</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;padding:0 18px;height:100%">
      <span style="color:rgba(255,255,255,0.18);font-size:9px;letter-spacing:.2em;text-transform:uppercase">Key</span>
      <select id="rootSel" style="${sel()}"></select>
      <select id="scaleSel" style="${sel()}"></select>
      <button id="vocBtn" style="${pill('#34d399')}">Vocals ON</button>
    </div>`;
  document.body.appendChild(bar);

  const nb=document.createElement('div');
  Object.assign(nb.style,{position:'fixed',bottom:'0',left:'0',width:'100%',height:'56px',background:'rgba(4,3,12,0.97)',backdropFilter:'blur(24px)',webkitBackdropFilter:'blur(24px)',borderTop:'1px solid rgba(255,255,255,0.05)',display:'flex',alignItems:'center',justifyContent:'center',gap:'28px',zIndex:'1000',fontFamily:'Inter,monospace'});
  nb.innerHTML=`
    <div style="display:flex;align-items:center;gap:6px">
      <span style="color:rgba(255,170,0,0.55);font-size:9px;letter-spacing:.18em;font-weight:500;text-transform:uppercase;margin-right:8px">L Bass</span>
      ${[0,1,2,3].map(i=>`<div id="ln${i}" style="${ntag()}">&nbsp;—&nbsp;</div>`).join('')}
    </div>
    <div style="width:1px;height:20px;background:rgba(255,255,255,0.05)"></div>
    <div style="display:flex;align-items:center;gap:6px">
      <span style="color:rgba(0,207,255,0.55);font-size:9px;letter-spacing:.18em;font-weight:500;text-transform:uppercase;margin-right:8px">R Melody</span>
      ${[0,1,2,3].map(i=>`<div id="rn${i}" style="${ntag()}">&nbsp;—&nbsp;</div>`).join('')}
    </div>`;
  document.body.appendChild(nb);

  document.getElementById('_upl').onchange=()=>loadTrack(document.getElementById('_upl').files[0]);
  const dz=document.getElementById('dropZone');
  dz.ondragover=e=>{e.preventDefault();dz.style.borderColor='rgba(103,232,249,0.35)';dz.style.background='rgba(103,232,249,0.04)';};
  dz.ondragleave=()=>{dz.style.borderColor='rgba(255,255,255,0.09)';dz.style.background='rgba(255,255,255,0.025)';};
  dz.ondrop=e=>{e.preventDefault();loadTrack(e.dataTransfer.files[0]);};
  document.getElementById('playBtn').onclick=()=>{ensureAudio();if(uploadedAudio){uploadedAudio.playbackRate=tempo;uploadedAudio.play().catch(()=>{});}};
  document.getElementById('pauseBtn').onclick=()=>{if(uploadedAudio)uploadedAudio.pause();};
  document.getElementById('stopBtn').onclick=()=>{if(uploadedAudio){uploadedAudio.pause();uploadedAudio.currentTime=0;}};
  document.getElementById('loopBtn').onclick=function(){
    if(uploadedAudio){uploadedAudio.loop=!uploadedAudio.loop;
      this.style.color=uploadedAudio.loop?'#67e8f9':'rgba(255,255,255,0.25)';
      this.style.borderColor=uploadedAudio.loop?'rgba(103,232,249,0.25)':'rgba(255,255,255,0.08)';}
  };
  document.getElementById('volSlider').oninput=function(){masterVol=this.value/100;if(uploadedAudio)uploadedAudio.volume=masterVol;document.getElementById('volVal').innerText=this.value+'%';};
  document.getElementById('tempoSlider').oninput=function(){tempo=this.value/100;if(uploadedAudio)uploadedAudio.playbackRate=tempo;document.getElementById('tempoVal').innerText=tempo.toFixed(2)+'×';};

  const rs=document.getElementById('rootSel');
  NOTE_NAMES.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.innerText=n;rs.appendChild(o);});
  rs.value=detRoot;
  rs.onchange=()=>{detRoot=+rs.value;rebuildNotes();updateKeyDisp();};

  const ss=document.getElementById('scaleSel');
  [['Min Penta','minor_penta'],['Maj Penta','major_penta'],['Minor','minor'],['Major','major']].forEach(([l,v])=>{
    const o=document.createElement('option');o.value=v;o.innerText=l;ss.appendChild(o);
  });
  ss.value='minor_penta';
  ss.onchange=()=>{detScale=ss.value;rebuildNotes();updateKeyDisp();};

  document.getElementById('vocBtn').onclick=function(){
    vocalOn=!vocalOn; applyVocalState();
    this.innerText=vocalOn?'Vocals OFF':'Vocals ON';
    this.style.color=vocalOn?'#f87171':'#34d399';
    this.style.borderColor=vocalOn?'rgba(248,113,113,0.3)':'rgba(52,211,153,0.3)';
  };
}

function tbtn(c){return`background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);color:${c};border-radius:7px;padding:6px 10px;cursor:pointer;font-family:Inter,monospace;font-size:13px;transition:all .15s;min-width:32px`;}
function sldr(){return`width:86px;accent-color:#67e8f9;cursor:pointer;background:transparent`;}
function sel(){return`background:rgba(255,255,255,0.03);color:#c084fc;border:1px solid rgba(192,132,252,0.18);border-radius:7px;padding:5px 8px;font-family:Inter,monospace;font-size:11px;cursor:pointer;outline:none`;}
function pill(c){return`background:rgba(255,255,255,0.02);color:${c};border:1px solid rgba(52,211,153,0.28);border-radius:20px;padding:5px 14px;cursor:pointer;font-family:Inter,monospace;font-size:10px;letter-spacing:.05em;white-space:nowrap;transition:all .2s`;}
function ntag(){return`background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:5px 10px;font-size:10px;color:rgba(255,255,255,0.28);font-family:Inter,monospace;letter-spacing:.04em;transition:all .15s`;}

function updateKeyDisp(){
  const kd=document.getElementById('keyDisp');
  if(kd) kd.innerText='KEY  '+NOTE_NAMES[detRoot]+' '+detScale.replace('_',' ');
  const rs=document.getElementById('rootSel'),ss=document.getElementById('scaleSel');
  if(rs) rs.value=detRoot; if(ss) ss.value=detScale;
}

// ── GLOVING COLORS ────────────────────────────────────────────────────────────
function jColor(h,k,alpha){
  const hue=(JOINT_HUES[k]+(h*120)+(beatP*30))%360;
  const s=0.95,l=0.58,c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs((hue/60)%2-1)),m=l-c/2;
  let r=0,g=0,b=0;
  if(hue<60){r=c;g=x;}else if(hue<120){r=x;g=c;}else if(hue<180){g=c;b=x;}
  else if(hue<240){g=x;b=c;}else if(hue<300){r=x;b=c;}else{r=c;b=x;}
  return color((r+m)*255,(g+m)*255,(b+m)*255,alpha);
}

// ── DRAW ──────────────────────────────────────────────────────────────────────
function draw(){
  analyzeAudio();

  // ghost-hand kill — use performance.now() consistently
  const now=performance.now();
  for(let h=0;h<2;h++) if(now-handLastSeen[h]>200) clearHand(h);

  const vid=document.getElementById('camFeed');

  if(chromaS>1.5&&vid&&vid.readyState>=2){
    push(); drawingContext.save();
    drawingContext.globalCompositeOperation='screen'; drawingContext.globalAlpha=.15;
    translate(width,0); scale(-1,1);
    tint(255,0,0,45); drawingContext.drawImage(vid,-chromaS*.28,0,width,height);
    tint(0,0,255,45); drawingContext.drawImage(vid,chromaS*.28,0,width,height);
    drawingContext.restore(); pop();
  }

  if(vid&&vid.readyState>=2){
    push(); translate(width,0); scale(-1,1);
    drawingContext.globalAlpha=1; drawingContext.drawImage(vid,0,0,width,height);
    pop();
  } else { background(8); }

  noStroke(); fill(0,0,0,map(beatP,0,1,18,2)); rect(0,0,width,height);
  if(flashA>1){ fill(255,30,180,flashA); rect(0,0,width,height); }

  // spectrum arc
  if(analyser){
    drawingContext.save(); drawingContext.globalCompositeOperation='screen';
    const bc=120, cx=width/2, cy=height+155, rad=height*.68;
    for(let i=0;i<bc;i++){
      const idx=floor(i/bc*dataArray.length*.5);
      const val=dataArray[idx]/255; if(val<.02) continue;
      const ang=map(i,0,bc,-PI*.73,-PI*.27)+PI;
      stroke(map(i,0,bc,255,0),map(i,0,bc,50,220),map(i,0,bc,180,255),map(val,0,1,18,130));
      strokeWeight(map(val,0,1,1,3));
      line(cx+cos(ang)*rad,cy+sin(ang)*rad,cx+cos(ang)*(rad+val*height*.28),cy+sin(ang)*(rad+val*height*.28));
    }
    noStroke(); drawingContext.restore();
  }

  // aura rings
  drawingContext.save(); drawingContext.globalCompositeOperation='screen';
  for(let b=0;b<8;b++){
    if(sBands[b]<.04) continue;
    noFill(); stroke(map(b,0,7,255,0),map(b,0,7,50,255),map(b,0,7,180,100),sBands[b]*34);
    strokeWeight(sBands[b]*3);
    circle(width/2,height/2,map(b,0,7,width*.05,width*.52)*sBands[b]*2);
  }
  drawingContext.restore();

  // beat rings
  if(bassP>.46){
    drawingContext.save(); drawingContext.globalCompositeOperation='screen'; noFill();
    for(let k=0;k<4;k++){
      stroke(255,30+k*35,180,map(bassP,.46,1,0,42)*(1-k*.24)); strokeWeight(2.5-k*.5);
      circle(width/2,height/2,map(bassP,.46,1,width*.08,width*.88)*(1+k*.14));
    }
    drawingContext.restore();
  }

  // effects — hard cap then draw
  if(ripples.length>CAP.r)   ripples.length=CAP.r;
  if(glows.length>CAP.g)     glows.length=CAP.g;
  if(particles.length>CAP.p) particles.length=CAP.p;
  if(shocks.length>CAP.s)    shocks.length=CAP.s;
  if(bursts.length>CAP.b)    bursts.length=CAP.b;

  drawingContext.save(); drawingContext.globalCompositeOperation='screen';
  for(let i=shocks.length-1;i>=0;i--){
    const s=shocks[i]; s.r+=s.spd; s.spd*=1.07; s.a-=2.4;
    if(s.a<=0){shocks.splice(i,1);continue;}
    noFill(); for(let t=0;t<4;t++){stroke(red(s.c),green(s.c),blue(s.c),s.a*(1-t*.24));strokeWeight(3.5-t*.7);circle(s.x,s.y,(s.r+t*16)*2);}
  }
  for(let i=bursts.length-1;i>=0;i--){
    const s=bursts[i]; s.life-=2.8; s.size+=s.grow;
    if(s.life<=0){bursts.splice(i,1);continue;}
    noStroke(); fill(red(s.c),green(s.c),blue(s.c),s.life*2.5); drawStar(s.x,s.y,s.size*.4,s.size,6);
  }
  for(let i=glows.length-1;i>=0;i--){
    const g=glows[i]; g.r+=g.spd*(1+beatP*.5); g.a-=g.fade;
    if(g.a<=0){glows.splice(i,1);continue;}
    noStroke();
    fill(red(g.c),green(g.c),blue(g.c),g.a*.7);  circle(g.x,g.y,g.r*2);
    fill(red(g.c),green(g.c),blue(g.c),g.a*.15); circle(g.x,g.y,g.r*4.5);
    fill(255,255,255,g.a*.04);                    circle(g.x,g.y,g.r*7);
  }
  for(let i=particles.length-1;i>=0;i--){
    const p=particles[i];
    p.x+=p.vx*(1+beatP*.3); p.y+=p.vy*(1+beatP*.3);
    p.vy+=.14; p.vx*=.97; p.life-=1.7;
    if(p.life<=0){particles.splice(i,1);continue;}
    noStroke(); fill(red(p.c),green(p.c),blue(p.c),p.life*2.2); circle(p.x,p.y,p.r*2);
  }
  for(let i=ripples.length-1;i>=0;i--){
    const rp=ripples[i]; rp.r+=rp.spd*(1+beatP*.4); rp.a-=1.5;
    if(rp.a<=0){ripples.splice(i,1);continue;}
    noFill(); stroke(red(rp.c),green(rp.c),blue(rp.c),rp.a);
    strokeWeight(map(rp.a,0,100,.3,2.5)); circle(rp.x,rp.y,rp.r*2);
  }
  drawingContext.restore();

  // hands
  drawingContext.save(); drawingContext.globalCompositeOperation='screen';
  for(let h=0;h<2;h++){
    if(!smoothLms[h]||smoothLms[h].length!==21) continue;
    const lms=smoothLms[h];
    const[r,g,b]=h===0?[255,50,180]:[100,70,255];
    const hc=color(r,g,b);

    handSize[h]=dist(lms[0].x,lms[0].y,lms[12].x,lms[12].y);
    const dep=constrain(map(handSize[h],50,280,.5,1.6),.3,2.0);
    const px=lms[9].x, py=lms[9].y;

    if(prevPalms[h]){
      const spd=dist(px,py,prevPalms[h].x,prevPalms[h].y);
      velocity[h]=lerp(velocity[h],spd,.3);
      if(spd>6)  glows.push({x:px,y:py,r:8,spd:3*(1+beatP),a:10+beatP*12,fade:1.5,c:hc});
      if(spd>18) for(let k=0;k<floor(map(spd,18,60,1,4));k++){
        const ang=random(TWO_PI), s2=random(1,spd*.08);
        particles.push({x:px,y:py,vx:cos(ang)*s2,vy:sin(ang)*s2,life:random(18,45),r:random(2,4),c:hc});
      }
    }
    prevPalms[h]={x:px,y:py};

    for(let k=0;k<21;k++){
      const kx=lms[k].x, ky=lms[k].y;
      const js=dist(kx,ky,prevJ[h][k].x,prevJ[h][k].y);
      jVel[h][k]=lerp(jVel[h][k],js,.35);
      if(js>20&&ALL_TIPS.includes(k)){
        const jc=jColor(h,k,220);
        bursts.push({x:kx,y:ky,size:random(5,12),grow:random(.5,1.5),life:random(22,50),c:jc});
        for(let rk=0;rk<floor(map(js,20,70,1,3));rk++)
          ripples.push({x:kx+random(-5,5),y:ky+random(-5,5),r:rk*5,spd:map(js,20,70,3,8)+rk,a:map(js,20,70,35,100),c:jc});
        glows.push({x:kx,y:ky,r:6,spd:3*(1+beatP),a:32+beatP*18,fade:2.5,c:jc});
      }
      prevJ[h][k]={x:kx,y:ky};
    }

    // pinch — raw distance trigger, instant response
    const tx0=lms[THUMB_TIP].x, ty0=lms[THUMB_TIP].y;
    for(let fi=0;fi<4;fi++){
      const ti=FINGER_TIPS[fi], tx=lms[ti].x, ty=lms[ti].y;
      const rawDist=handSize[h]>10?dist(tx,ty,tx0,ty0)/handSize[h]:1;
      pDist[h][fi]=lerp(pDist[h][fi],rawDist,.55);
      const isP=rawDist<PINCH_T;

      if(isP&&!pState[h][fi]&&pCool[h][fi]===0){
        playNote(fi,h,0.9);
        const fc=color(FINGER_COLS[fi]);
        const mx=(tx+tx0)/2, my=(ty+ty0)/2;
        glows.push({x:mx,y:my,r:20,spd:7*(1+beatP),a:120,fade:4,c:fc});
        glows.push({x:mx,y:my,r:7,spd:14,a:90,fade:5,c:color(255,255,255)});
        for(let k=0;k<5;k++) ripples.push({x:mx,y:my,r:k*8,spd:4+k*2+beatP*3,a:105,c:fc});
        shocks.push({x:mx,y:my,r:10,spd:7+beatP*5,a:90,c:fc});
        for(let k=0;k<12;k++){const ang=random(TWO_PI),s=random(2,9+beatP*4);particles.push({x:mx,y:my,vx:cos(ang)*s,vy:sin(ang)*s-2,life:random(35,85),r:random(2,6),c:fc});}
        for(let k=0;k<3;k++) bursts.push({x:mx+random(-15,15),y:my+random(-15,15),size:random(7,16),grow:random(.7,1.8),life:random(28,58),c:fc});
        const el=document.getElementById((h===0?'ln':'rn')+fi);
        if(el){
          el.style.background=FINGER_COLS[fi]+'30'; el.style.color='#fff'; el.style.borderColor=FINGER_COLS[fi]+'80';
          setTimeout(()=>{el.style.background='rgba(255,255,255,0.025)';el.style.color='rgba(255,255,255,0.28)';el.style.borderColor='rgba(255,255,255,0.06)';},400);
        }
        pCool[h][fi]=6;
      }
      if(pCool[h][fi]>0) pCool[h][fi]--;
      pState[h][fi]=isP;

      const ratio=map(constrain(pDist[h][fi],0,PINCH_T*2),0,PINCH_T*2,1,0);
      if(ratio>.05){
        const fc=color(FINGER_COLS[fi]);
        stroke(red(fc),green(fc),blue(fc),ratio*142*(1+beatP*.5)); strokeWeight(ratio*3);
        line(tx,ty,tx0,ty0);
      }
    }

    // skeleton
    const bM=1+beatP*.55;
    const conn=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
    for(const[a,bb]of conn){
      stroke(r,g,b,16*dep*bM); strokeWeight(7*dep); line(lms[a].x,lms[a].y,lms[bb].x,lms[bb].y);
      stroke(r,g,b,140*dep*bM); strokeWeight(1.5*dep); line(lms[a].x,lms[a].y,lms[bb].x,lms[bb].y);
    }

    // joints
    const tipSet=new Set([...FINGER_TIPS,THUMB_TIP]);
    for(let k=0;k<21;k++){
      const kx=lms[k].x, ky=lms[k].y;
      const it=tipSet.has(k), jv=jVel[h][k];
      const base=(it?map(velocity[h],0,40,12,36):map(velocity[h],0,40,4,14))*dep*(1+beatP*.44);
      const jc=jColor(h,k,255);
      noStroke();
      fill(red(jc),green(jc),blue(jc),7+jv*.4);  circle(kx,ky,base*6);
      fill(red(jc),green(jc),blue(jc),32+jv);     circle(kx,ky,base*2.8);
      fill(red(jc),green(jc),blue(jc),200);        circle(kx,ky,base*.85);
      fill(255,255,255,255);                        circle(kx,ky,base*.3);
      if(jv>14&&it){ fill(red(jc),green(jc),blue(jc),map(jv,14,50,12,65)); circle(kx,ky,base*4); }
      const fi2=FINGER_TIPS.indexOf(k);
      if(fi2>=0&&pState[h][fi2]){
        const fc=color(FINGER_COLS[fi2]);
        fill(red(fc),green(fc),blue(fc),205); circle(kx,ky,base*2.8);
        fill(255,255,255,170); circle(kx,ky,base);
      }
    }
  }
  drawingContext.restore();

  noStroke(); fill(255,255,255,8); textSize(11); textAlign(RIGHT);
  text('LOOP IT',width-20,height/2);
  if(millis()<7000){
    noStroke(); fill(255,255,255,map(millis(),5000,7000,100,0));
    textSize(11); textAlign(CENTER);
    text('upload a track  ·  key auto-detects  ·  pinch fingers to play in-key notes',width/2,height-70);
  }
}

function drawStar(x,y,r1,r2,pts){
  const a=TWO_PI/pts,h=a/2; beginShape();
  for(let i=-HALF_PI;i<TWO_PI-HALF_PI;i+=a){ vertex(x+cos(i)*r2,y+sin(i)*r2); vertex(x+cos(i+h)*r1,y+sin(i+h)*r1); }
  endShape(CLOSE);
}

function windowResized(){ resizeCanvas(windowWidth,windowHeight); }
