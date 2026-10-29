import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import os
import time

st.set_page_config(
    page_title="Virtual Mouse AI | Youssef",
    page_icon="🖐️",
    layout="wide"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081426, #102943);
    color: white;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    text-align: center;
    padding: 25px;
    border: 1px solid #244d70;
    border-radius: 20px;
    background: linear-gradient(120deg,#102943,#153c58);
}
.hero h1 {
    color: #53d8ed;
    font-size: 45px;
}
.hero h2 {
    color: white;
    font-size: 25px;
}
.hero p {
    color: #c5d9e8;
    font-size: 17px;
}
.marquee {
    overflow: hidden;
    white-space: nowrap;
    color: #53d8ed;
    font-size: 17px;
    margin: 20px 0;
}
.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: move 15s linear infinite;
}
@keyframes move {
    from { transform: translateX(0); }
    to { transform: translateX(-100%); }
}
.panel {
    background: #122b43;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #244d70;
    margin-bottom: 15px;
}
div.stButton > button {
    background: #16a9c5;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #087e9d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h2>Engineer Youssef Khaled Madin</h2>
    <h1>VIRTUAL MOUSE AI</h1>
    <p>Control your computer using hand gestures</p>
</div>

<div class="marquee">
    <span>
    ARTIFICIAL INTELLIGENCE • COMPUTER VISION •
    MEDIAPIPE • HAND TRACKING • PYTHON •
    VIRTUAL MOUSE •
    </span>
</div>
""", unsafe_allow_html=True)

# ---------- INTRO ----------
st.markdown("## 🖐️ Welcome to Virtual Mouse AI")

st.write("""
This project uses Artificial Intelligence and Computer Vision
to detect hand landmarks and recognize hand gestures.

The goal is to make computer interaction easier using hand movements.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="panel">🖐️<h3>Hand Detection</h3><p>Detect hand landmarks.</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">🖱️<h3>Mouse Control</h3><p>Move the cursor using your hand.</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="panel">👆<h3>Gesture Click</h3><p>Recognize finger gestures.</p></div>', unsafe_allow_html=True)

st.divider()

# ---------- CAMERA DEMO ----------
st.markdown("## 📷 Try Hand Detection")

st.info(
    "Allow camera access, take a picture of your hand, "
    "and the AI will detect its landmarks."
)

model_path = "hand_landmarker.task"

if os.path.exists(model_path):

    picture = st.camera_input("Take a picture of your hand")

    if picture is not None:

        image = Image.open(picture).convert("RGB")
        frame = np.array(image)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        Options = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = Options(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=1
        )

        with HandLandmarker.create_from_options(options) as detector:

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:

                height, width, _ = frame.shape

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]

                for hand in result.hand_landmarks:

                    for start, end in connections:

                        x1 = int(hand[start].x * width)
                        y1 = int(hand[start].y * height)
                        x2 = int(hand[end].x * width)
                        y2 = int(hand[end].y * height)

                        cv2.line(
                            frame, (x1, y1), (x2, y2),
                            (0, 220, 255), 3
                        )

                    for point in hand:

                        x = int(point.x * width)
                        y = int(point.y * height)

                        cv2.circle(
                            frame, (x, y), 6,
                            (255, 80, 40), -1
                        )

                st.success("Hand detected successfully!")
                st.image(frame, caption="AI Hand Detection", use_container_width=True)

            else:
                st.warning("No hand detected. Try again.")

else:
    st.warning(
        "Model file not found. Add hand_landmarker.task "
        "to the same folder as app.py."
    )

st.divider()

# ---------- DOWNLOAD ----------
st.markdown("## 📥 Download the Project")

st.write("""
Download the local version to use the actual mouse-control
features on your own computer.
""")

if os.path.exists("virtual_mouse.py"):

    with open("virtual_mouse.py", "rb") as file:
        st.download_button(
            "Download Virtual Mouse Code",
            data=file,
            file_name="virtual_mouse.py",
            mime="text/x-python"
        )

if os.path.exists(model_path):

    with open(model_path, "rb") as file:
        st.download_button(
            "Download Hand Model",
            data=file,
            file_name="hand_landmarker.task"
        )

st.divider()

st.markdown("""
<div style="text-align:center; color:#8daabd; padding:20px;">
    <h3>Engineer Youssef Khaled Madin</h3>
    <p>Built with Python • Streamlit • MediaPipe • OpenCV</p>
</div>
""", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#8daabd; padding:20px;">
        <h3>Engineer Youssef Khaled Madin</h3>
        <p>Built with Python • Streamlit • MediaPipe</p>
    </div>
    """,
    unsafe_allow_html=True,
)
