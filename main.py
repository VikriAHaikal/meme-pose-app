import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. UI MOBILE-OPTIMIZED ---
st.set_page_config(page_title="Meme AI Pro", layout="centered")
st.markdown(
    "<style>.stVideo, video { width: 100% !important; border-radius: 15px; }</style>", unsafe_allow_html=True)

st.title("🎭 Meme AI Pose Pro")

# --- 2. CACHE DETECTOR (Biar nggak berat) ---


@st.cache_resource
def get_detectors():
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    h_det = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)
    f_det = mp_face.FaceMesh(refine_landmarks=False,
                             min_detection_confidence=0.5)
    return h_det, f_det


HANDS_DET, FACE_DET = get_detectors()

# --- 3. LOGIKA PROCESSOR ---


class SafeProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pose = None
        self.frame_skip = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self.frame_skip += 1
        if self.frame_skip % 4 == 0:  # Deteksi tiap 4 frame saja biar nggak slowmo
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_hands = HANDS_DET.process(rgb)

            # Deteksi Tangan Sederhana dulu
            if res_hands.multi_hand_landmarks:
                hlm = res_hands.multi_hand_landmarks[0].landmark
                if hlm[8].y < hlm[6].y:
                    self.last_pose = "pointing"
                else:
                    self.last_pose = None
            else:
                self.last_pose = None

        if self.last_pose == "pointing":
            cv2.putText(img, "POINTING!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# --- 4. START STREAMER (Koneksi diperkuat) ---
st.info("Klik START. Jika browser minta izin kamera, klik ALLOW.")

webrtc_streamer(
    key="meme-pro-v5",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SafeProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]}
        ]
    },
    media_stream_constraints={
        "video": {"facingMode": "user", "width": {"ideal": 480}, "height": {"ideal": 360}},
        "audio": False
    },
    async_processing=True,
)
