import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Virtual Mouse AI | Youssef",
    page_icon="🖐️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#081426,#102943); color:#fff; }
    .block-container { max-width:1200px; padding-top:1.5rem; }
    .hero {
        text-align:center; padding:24px; border:1px solid #244d70;
        border-radius:20px; background:linear-gradient(120deg,#102943,#153c58);
    }
    .hero h1 { color:#53d8ed; font-size:42px; margin-bottom:8px; }
    .hero h2 { color:white; font-size:23px; }
    .hero p { color:#c5d9e8; }
    .feature {
        background:#122b43; border:1px solid #244d70; border-radius:14px;
        padding:18px; min-height:125px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2>Engineer Youssef Khaled Madin</h2>
      <h1>VIRTUAL MOUSE AI</h1>
      <p>Real-time hand tracking and gesture interaction in your browser</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="feature">🖐️ <b>21 Hand Landmarks</b><p>Detect and draw hand keypoints live.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="feature">🖱️ <b>Virtual Cursor</b><p>Move the on-screen pointer with your index finger.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="feature">👆 <b>Gesture Click</b><p>Pinch thumb and index finger to click demo controls.</p></div>', unsafe_allow_html=True)

st.markdown("## 📷 Live camera experience")
st.info(
    "Allow camera access when your browser asks. Keep your hand visible. "
    "Move your index finger to move the virtual cursor; pinch thumb and index finger to click the test buttons below."
)

# MediaPipe runs in the visitor's browser. This avoids sending camera frames to
# the Streamlit server and works on HTTPS-hosted pages where camera permission is allowed.
html = r"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing:border-box; }
  body { margin:0; font-family:Arial,sans-serif; color:#eaf6ff; background:#0b1c30; }
  .wrap { display:grid; grid-template-columns:minmax(0, 1.5fr) minmax(220px, .8fr); gap:14px; }
  .camera, .side { background:#122b43; border:1px solid #28516f; border-radius:14px; padding:12px; }
  .stage { position:relative; width:100%; aspect-ratio:4/3; background:#07111e; border-radius:10px; overflow:hidden; }
  video, canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }
  #overlay { pointer-events:none; }
  .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  button { border:0; border-radius:9px; padding:11px 14px; font-weight:bold; cursor:pointer; color:white; background:#119dbb; }
  button.secondary { background:#31516b; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  .status { margin-top:10px; padding:10px; border-radius:9px; background:#0b2035; font-size:14px; }
  .demo { margin-top:12px; padding:14px; background:#0b2035; border-radius:10px; }
  .demo button { margin:5px 5px 0 0; }
  #cursor { position:fixed; width:22px; height:22px; border:3px solid #53e6ff; border-radius:50%; z-index:20; pointer-events:none; transform:translate(-50%,-50%); display:none; box-shadow:0 0 12px #53e6ff; }
  #cursor.pinching { background:#53e6ff88; border-color:#ffb84d; }
  .small { color:#a9c5d9; font-size:13px; line-height:1.5; }
  @media(max-width:700px) { .wrap { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <section class="camera">
    <div class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
    </div>
    <div class="actions">
      <button id="start">Start camera</button>
      <button id="stop" class="secondary" disabled>Stop camera</button>
    </div>
    <div id="status" class="status">Camera is off. Press “Start camera”.</div>
  </section>
  <aside class="side">
    <h3 style="margin-top:4px">How to use</h3>
    <p class="small">1. Allow camera access.<br>2. Show one hand clearly.<br>3. Move your index fingertip to move the virtual cursor.<br>4. Pinch your thumb and index finger to click a test button.</p>
    <div class="demo">
      <b>Try the virtual mouse here</b><br>
      <button class="target" id="test1">Test button 1</button>
      <button class="target" id="test2">Test button 2</button>
      <div id="result" class="small" style="margin-top:10px">No click yet.</div>
    </div>
    <p class="small">Privacy: camera processing happens in this browser frame; video is not uploaded to the Streamlit app by this code.</p>
    <p class="small">Note: the virtual cursor and clicks operate inside this demo area, not on other tabs or desktop applications.</p>
  </aside>
</div>
<div id="cursor"></div>

<script type="module">
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const statusBox = document.getElementById('status');
const cursor = document.getElementById('cursor');
let stream = null, landmarker = null, running = false, raf = null;
let lastClick = 0, wasPinching = false, smoothX = null, smoothY = null;
const modelURL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const visionURL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const connections = [
 [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
 [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],
 [18,19],[19,20],[0,17]
];

function resizeCanvas() {
  const r = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width*dpr));
  canvas.height = Math.max(1, Math.round(r.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function drawHand(points) {
  const w = video.clientWidth, h = video.clientHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 3; ctx.strokeStyle = "#32e6ff";
  for (const [a,b] of connections) {
    ctx.beginPath(); ctx.moveTo(points[a].x*w, points[a].y*h);
    ctx.lineTo(points[b].x*w, points[b].y*h); ctx.stroke();
  }
  for (const p of points) {
    ctx.beginPath(); ctx.arc(p.x*w,p.y*h,4,0,Math.PI*2);
    ctx.fillStyle="#ff784e"; ctx.fill();
  }
}
function handlePinch(points) {
  const dx=points[4].x-points[8].x, dy=points[4].y-points[8].y;
  const pinching=Math.hypot(dx,dy)<0.055;
  cursor.classList.toggle("pinching",pinching);
  const now=Date.now();
  if(pinching && !wasPinching && now-lastClick>650) {
    const el=document.elementFromPoint(cursor.getBoundingClientRect().left+11,cursor.getBoundingClientRect().top+11);
    if(el && el.closest(".target")) {
      el.closest(".target").click();
      lastClick=now;
    }
  }
  wasPinching=pinching;
}
function loop() {
  if(!running) return;
  if(video.readyState >= 2 && landmarker) {
    const result=landmarker.detectForVideo(video, performance.now());
    if(result.landmarks && result.landmarks.length) {
      const points=result.landmarks[0];
      drawHand(points);
      const p=points[8];
      // The video is mirrored; reverse x to match the displayed hand.
      const x=(1-p.x)*window.innerWidth;
      const y=p.y*window.innerHeight;
      smoothX = smoothX===null ? x : smoothX + (x-smoothX)*0.28;
      smoothY = smoothY===null ? y : smoothY + (y-smoothY)*0.28;
      cursor.style.left=smoothX+"px"; cursor.style.top=smoothY+"px"; cursor.style.display="block";
      handlePinch(points);
      statusBox.textContent=wasPinching ? "Pinch detected — click gesture" : "Hand detected — move your index finger";
    } else {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      cursor.style.display="none"; wasPinching=false;
      statusBox.textContent="No hand detected. Keep your hand visible.";
    }
  }
  raf=requestAnimationFrame(loop);
}
startBtn.addEventListener("click", async () => {
  try {
    statusBox.textContent="Loading hand tracking model…";
    startBtn.disabled=true;
    const vision=await import(visionURL);
    const files=await vision.FilesetResolver.forVisionTasks(visionURL+"/wasm");
    landmarker=await vision.HandLandmarker.createFromOptions(files,{
      baseOptions:{modelAssetPath:modelURL, delegate:"GPU"},
      runningMode:"VIDEO", numHands:1,
      minHandDetectionConfidence:0.55,
      minHandPresenceConfidence:0.55,
      minTrackingConfidence:0.55
    });
    stream=await navigator.mediaDevices.getUserMedia({
      audio:false,
      video:{facingMode:"user", width:{ideal:1280}, height:{ideal:720}}
    });
    video.srcObject=stream;
    await video.play();
    resizeCanvas();
    running=true; stopBtn.disabled=false;
    statusBox.textContent="Camera started. Show your hand.";
    loop();
  } catch(err) {
    startBtn.disabled=false;
    statusBox.textContent="Could not start camera/tracking: "+err.message+
      ". Check browser camera permission and use HTTPS (or localhost).";
  }
});
stopBtn.addEventListener("click", () => {
  running=false;
  if(raf) cancelAnimationFrame(raf);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  stream=null; video.srcObject=null;
  if(landmarker) { landmarker.close(); landmarker=null; }
  ctx.clearRect(0,0,canvas.width,canvas.height);
  cursor.style.display="none";
  startBtn.disabled=false; stopBtn.disabled=true;
  statusBox.textContent="Camera stopped.";
});
window.addEventListener("resize",resizeCanvas);
document.getElementById("test1").addEventListener("click",()=>document.getElementById("result").textContent="Great! Test button 1 clicked using the virtual mouse.");
document.getElementById("test2").addEventListener("click",()=>document.getElementById("result").textContent="Great! Test button 2 clicked using the virtual mouse.");
</script>
</body>
</html>
"""

components.html(html, height=650, scrolling=False)

st.markdown(
    """
    <div style="text-align:center;color:#8daabd;padding:18px;">
      <h3>Engineer Youssef Khaled Madin</h3>
      <p>Built with Python • Streamlit • MediaPipe • Computer Vision</p>
    </div>
    """,
    unsafe_allow_html=True,
)
