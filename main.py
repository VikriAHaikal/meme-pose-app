import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. SETTING HALAMAN & MOBILE CSS ---
st.set_page_config(page_title="Meme AI Pro", layout="centered")

# Inject CSS agar video responsif di layar HP
st.markdown("""
    <style>
    .element-container img, .stVideo {
        width: 100% !important;
        height: auto !important;
        border-radius: 10px;
    }
    canvas {
        max-width: 100% !important;
        height: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Meme AI Pose Mobile")

# --- 2. KONFIGURASI ASSETS ---
ASSETS_PATH = "assets"
MEME_FILES = {
    "pointing": "monkey_pointing.png",
    "thinking": "monkey_thinking.png",
    "surprised": "monkey_surprised.png",
    "wink": "monkey_wink.png"
}
VIDEO_FILE = "prabowo_video.mp4"

# Fungsi muat assets dengan proteksi error
def load_assets():
    loaded = {}
    for key, name in MEME_FILES.items():
        p = os.path.join(ASSETS_PATH, name)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                loaded[key] = img
    return loaded

AVAILABLE_MEMES = load_assets()
VIDEO_PATH = os.path.join(ASSETS_PATH, VIDEO_FILE)

# --- 3. LOGIKA AI (MEDIAPIPE) ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

class MobileMemeProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Resizing Input agar ringan di HP (360p)
        h, w = img.shape[:2]
        target_w = 480
        target_h = int(h * (target_w / w))
        img = cv2.resize(img, (target_w, target_h))
        
        # Proses AI
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_hands = hands_detector.process(rgb)

        pose = None
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            # Deteksi Pointing
            if hlm[8].y < hlm[6].y: 
                pose = "pointing"
            # Deteksi Fist (Prabowo)
            if all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]) and hlm[0].y < 0.5:
                pose = "video"

        # Gabungkan Atas (Kamera) & Bawah (Meme)
        # Ukuran kanvas jadi 480 x (target_h * 2)
        canvas = np.zeros((target_h * 2, target_w, 3), dtype=np.uint8)
        canvas[0:target_h, 0:target_w] = img
        
        meme_area = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        if pose == "video" and self.video_cap:
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            if ret: 
                meme_area = cv2.resize(v_f, (target_w, target_h))
        elif pose in AVAILABLE_MEMES:
            meme_area = cv2.resize(AVAILABLE_MEMES[pose], (target_w, target_h))

        canvas[target_h:target_h*2, 0:target_w] = meme_area
        return frame.from_ndarray(canvas, format="bgr24")

# --- 4. RUNNER ---
st.info("💡 Tips: Gunakan mode Portrait. Klik START dan tunggu lampu kamera menyala.")

webrtc_streamer(
    key="mobile-meme-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MobileMemeProcessor,
    # Ice Servers Google agar tembus blokir jaringan HP
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    },
    # Resolusi rendah agar stabil di 4G/5G
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True,
)

if not AVAILABLE_MEMES:
    st.warning("⚠️ Folder 'assets' belum lengkap atau nama file salah di GitHub.")