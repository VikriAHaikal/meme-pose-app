import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. SETTING MOBILE UI & CSS ---
st.set_page_config(page_title="Meme AI Mobile", layout="centered")

# CSS untuk memastikan video memenuhi lebar HP tapi tidak melebihi tinggi layar
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        text-align: center;
    }
    .stVideo, video {
        width: 100% !important;
        max-height: 70vh !important; /* Batasi tinggi video agar muat di layar HP */
        border-radius: 15px;
        border: 2px solid #ff4b4b;
    }
    .stAlert {
        padding: 0.5rem !important;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Meme AI Pose Pro")

# --- 2. KONFIGURASI ASSETS ---
ASSETS_PATH = "assets"
MEME_FILES = {
    "pointing": "monkey_pointing.png",
    "thinking": "monkey_thinking.png",
    "surprised": "monkey_surprised.png",
    "wink": "monkey_wink.png"
}
VIDEO_FILE = "prabowo_video.mp4"


def load_memes():
    loaded = {}
    for key, name in MEME_FILES.items():
        p = os.path.join(ASSETS_PATH, name)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                loaded[key] = img
    return loaded


AVAILABLE_MEMES = load_memes()
VIDEO_PATH = os.path.join(ASSETS_PATH, VIDEO_FILE)

# --- 3. LOGIKA AI (MEDIAPIPE) ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


class MobileOverlayProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture(
            VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror agar natural bagi pengguna
        h, w, _ = img.shape

        # Proses AI
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_hands = hands_detector.process(rgb)

        pose = None
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            # Pointing
            if hlm[8].y < hlm[6].y:
                pose = "pointing"
            # Fist (Prabowo)
            if all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]) and hlm[0].y < 0.5:
                pose = "video"

        # --- LOGIKA OVERLAY (PIP) ---
        # Ukuran Overlay Meme (25% dari lebar video)
        pip_w = int(w * 0.35)
        pip_h = int(h * 0.35)

        meme_img = None
        if pose == "video" and self.video_cap:
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            if ret:
                meme_img = v_f
        elif pose in AVAILABLE_MEMES:
            meme_img = AVAILABLE_MEMES[pose]

        if meme_img is not None:
            meme_res = cv2.resize(meme_img, (pip_w, pip_h))
            # Tempel di pojok kanan bawah dengan sedikit margin
            margin = 10
            overlay = img.copy()
            # Buat background putih sedikit transparan di belakang meme (Border effect)
            cv2.rectangle(img, (w-pip_w-margin-2, h-pip_h-margin-2),
                          (w-margin+2, h-margin+2), (255, 255, 255), -1)
            img[h-pip_h-margin:h-margin, w-pip_w-margin:w-margin] = meme_res

            # Tambahkan teks label pose
            label = "PRABOWO MODE" if pose == "video" else f"POSE: {pose.upper()}"
            cv2.putText(img, label, (w-pip_w-margin, h-pip_h-margin-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# --- 4. TAMPILAN UTAMA ---
st.info("📱 **Mode Mobile**: Pastikan wajah dan tangan terlihat di kamera.")

webrtc_streamer(
    key="mobile-overlay",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MobileOverlayProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {
            "facingMode": "user",  # Paksa pakai kamera depan
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    async_processing=True,
)

# Sidebar untuk panduan agar tidak memenuhi layar utama HP
with st.sidebar:
    st.subheader("📸 Panduan Pose")
    st.write("- **Tunjuk Jari**: Muncul monyet menunjuk.")
    st.write("- **Kepal Tangan**: Muncul video Pak Prabowo.")
    if not AVAILABLE_MEMES:
        st.error("⚠️ File meme di folder 'assets' belum lengkap!")
