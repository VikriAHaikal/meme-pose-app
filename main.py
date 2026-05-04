import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os

# --- 1. AUTO-CHECK ASSETS (Biar nggak tebak-tebak buah manggis) ---
st.set_page_config(page_title="Meme AI Debugger", layout="centered")
st.title("🎭 Meme AI Pose - Final Check")

ASSETS_PATH = "assets/"
files_to_check = {
    "pointing": "monkey_pointing.png",
    "thinking": "monkey_thinking.jpg",
    "surprised": "monkey_surprised.png",
    "wink": "monkey_wink.png",
    "video": "prabowo_video.mp4"
}

available_memes = {}
st.sidebar.header("📁 Status Assets")

for key, name in files_to_check.items():
    full_path = os.path.join(ASSETS_PATH, name)
    if os.path.exists(full_path):
        st.sidebar.success(f"✅ {name} ditemukan")
        if key != "video":
            available_memes[key] = cv2.imread(full_path)
    else:
        st.sidebar.error(f"❌ {name} TIDAK ADA")

# --- 2. LOGIKA DETEKSI (SUPER RINGAN) ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


class MemeProcessor(VideoProcessorBase):
    def __init__(self):
        video_path = os.path.join(ASSETS_PATH, files_to_check["video"])
        self.video_cap = cv2.VideoCapture(
            video_path) if os.path.exists(video_path) else None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        # Deteksi Tangan Saja (Biar nggak berat di HP)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_hands = hands_detector.process(rgb)

        pose = None
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            # Pointing
            if hlm[8].y < hlm[6].y:
                pose = "pointing"
            # Fist (Prabowo)
            if all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]):
                pose = "video"

        # Gabungkan Atas (Kamera) & Bawah (Meme)
        canvas = np.zeros((h * 2, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = img

        meme_area = np.zeros((h, w, 3), dtype=np.uint8)
        if pose == "video" and self.video_cap:
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            if ret:
                meme_area = cv2.resize(v_f, (w, h))
        elif pose in available_memes:
            meme_area = cv2.resize(available_memes[pose], (w, h))

        canvas[h:h*2, 0:w] = meme_area
        return frame.from_ndarray(canvas, format="bgr24")


# --- 3. JALANKAN WEBRTC ---
webrtc_streamer(
    key="meme-final-v2",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MemeProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
