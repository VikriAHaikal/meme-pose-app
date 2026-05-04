import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os

# --- CEK FILE ASSETS ---


def check_assets():
    files = ["monkey_pointing.png", "monkey_thinking.jpg",
             "monkey_surprised.png", "monkey_wink.png", "prabowo_video.mp4"]
    for f in files:
        if not os.path.exists(f"assets/{f}"):
            st.error(f"⚠️ File assets/{f} tidak ditemukan di GitHub!")


# Import Mediapipe
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


class MemeProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture("assets/prabowo_video.mp4")
        # Load memes sekali saja di awal
        self.memes = {
            "pointing": cv2.imread("assets/monkey_pointing.png"),
            "thinking": cv2.imread("assets/monkey_thinking.jpg")
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror
        h, w, c = img.shape

        # Proses deteksi (Sederhana dulu untuk tes)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_hands = hands_detector.process(rgb)

        pose = None
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            # Pointing: Telunjuk (8) lebih tinggi dari tekukannya (6)
            if hlm[8].y < hlm[6].y:
                pose = "pointing"
            # Prabowo (Fist): Semua ujung jari lebih rendah dari tekukannya
            if all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]) and hlm[0].y < 0.5:
                pose = "prabowo"

        # Buat Canvas Vertical (Atas: Kamera, Bawah: Meme)
        canvas = np.zeros((h * 2, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = img

        meme_area = np.zeros((h, w, 3), dtype=np.uint8)
        if pose == "prabowo":
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            if ret:
                meme_area = cv2.resize(v_f, (w, h))
        elif pose in self.memes and self.memes[pose] is not None:
            meme_area = cv2.resize(self.memes[pose], (w, h))

        canvas[h:h*2, 0:w] = meme_area
        return frame.from_ndarray(canvas, format="bgr24")


st.title("🎭 Meme AI Pro - Live")
check_assets()

# Konfigurasi WebRTC yang paling stabil untuk Cloud
webrtc_streamer(
    key="meme-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MemeProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,  # Penting agar tidak lag
)
