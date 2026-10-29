import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Virtual Mouse AI | Youssef",
    page_icon="🖐️",
    layout="wide",
)

# Use the folder containing app.py, not the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

# ---------------- DESIGN ----------------
st.markdown(
    """
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
        background: linear-gradient(120deg, #102943, #153c58);
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
        min-height: 145px;
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
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h2>Engineer Youssef Khaled Madin</h2>
        <h1>VIRTUAL MOUSE AI</h1>
        <p>Control your computer using hand gestures</p>
    </div>
    <div class="marquee">
        <span>
        ARTIFICIAL INTELLIGENCE • COMPUTER VISION •
        MEDIAPIPE • HAND TRACKING • PYTHON • VIRTUAL MOUSE •
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- INTRO ----------------
st.markdown("## 🖐️ Welcome to Virtual Mouse AI")
st.write(
    """
    This project uses Artificial Intelligence and Computer Vision
    to detect hand landmarks and recognize hand gestures.

    The goal is to make computer interaction easier using hand movements.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="panel">🖐️<h3>Hand Detection</h3>'
        '<p>Detect hand landmarks.</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="panel">🖱️<h3>Mouse Control</h3>'
        '<p>Move the cursor using your hand.</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="panel">👆<h3>Gesture Click</h3>'
        '<p>Recognize finger gestures.</p></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------- CAMERA DEMO ----------------
st.markdown("## 📷 Try Hand Detection")
st.info(
    "Use the camera control below, allow camera permission in your browser, "
    "and take a picture of your hand. The AI will detect its landmarks."
)

if not MODEL_PATH.is_file():
    st.error(
        "The hand model file was not found. Make sure "
        "`hand_landmarker.task` is in the same folder as `app.py`."
    )
else:
    picture = st.camera_input("Take a picture of your hand")

    if picture is not None:
        try:
            # Load the photo as RGB data for MediaPipe.
            image = Image.open(picture).convert("RGB")
            frame = np.asarray(image, dtype=np.uint8)
            frame = np.ascontiguousarray(frame)

            # Import only when the user submits a camera picture.
            import mediapipe as mp

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
            )

            with HandLandmarker.create_from_options(options) as detector:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame,
                )
                result = detector.detect(mp_image)

            if result.hand_landmarks:
                output_image = image.copy()
                draw = ImageDraw.Draw(output_image)
                width, height = output_image.size

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]

                for hand in result.hand_landmarks:
                    points = [
                        (int(point.x * width), int(point.y * height))
                        for point in hand
                    ]

                    for start, end in connections:
                        draw.line(
                            [points[start], points[end]],
                            fill=(0, 220, 255),
                            width=5,
                        )

                    for x, y in points:
                        radius = 6
                        draw.ellipse(
                            (x - radius, y - radius, x + radius, y + radius),
                            fill=(255, 80, 40),
                        )

                st.success("Hand detected successfully!")
                st.image(
                    output_image,
                    caption="AI Hand Detection",
                    use_container_width=True,
                )
            else:
                st.warning("No hand detected. Keep your hand clearly visible and try again.")

        except Exception as error:
            st.error("The camera image was received, but hand detection failed.")
            st.exception(error)

st.divider()

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div style="text-align:center; color:#8daabd; padding:20px;">
        <h3>Engineer Youssef Khaled Madin</h3>
        <p>Built with Python • Streamlit • MediaPipe</p>
    </div>
    """,
    unsafe_allow_html=True,
)
