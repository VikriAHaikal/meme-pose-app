import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. SETTING UI MOBILE ---
st.set_page_config(page_title="Meme AI Pro", layout="centered")
st.markdown(
    "<style>.stVideo, video { width: 100% !important; border-radius: 15px; }</style>", unsafe_allow_html=True)
st.title("🎭 Meme AI Video Filter")

# --- 2. LOAD ASSETS & DETECTOR ---


@st.cache_resource
def load_resources():
    # Load Detector (Complexity 0 = Paling Cepat)
    h_det = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)
    f_det = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=False, min_detection_confidence=0.5)

    # Load Memes
    memes = {}
    path = "assets"
    files = {"pointing": "monkey_pointing.png", "thinking": "monkey_thinking.png",
             "surprised": "monkey_surprised.png", "wink": "monkey_wink.png"}
    for k, v in files.items():
        p = os.path.join(path, v)
        if not os.path.exists(p):
            p = p.replace(".png", ".jpg")
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                memes[k] = img
    return h_det, f_det, memes


HANDS, FACE, AVAILABLE_MEMES = load_resources()
VIDEO_PRABOWO = "assets/prabowo_video.mp4"

# --- 3. CORE PROCESSOR ---


class FunnyVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.cap = cv2.VideoCapture(
            VIDEO_PRABOWO) if os.path.exists(VIDEO_PRABOWO) else None
        self.last_pose = None
        self.count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # FRAME SKIPPING (Hanya proses AI tiap 4 frame biar lancar/gak slowmo)
        self.count += 1
        if self.count % 4 == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_h = HANDS.process(rgb)
            res_f = FACE.process(rgb)

            pose = None
            if res_f.multi_face_landmarks:
                flm = res_f.multi_face_landmarks[0].landmark
                # Kaget (Mulut mangap)
                if (flm[14].y - flm[13].y) > 0.06:
                    pose = "surprised"

            if res_h.multi_hand_landmarks and not pose:
                hlm = res_h.multi_hand_landmarks[0].landmark
                # Pointing (Telunjuk)
                if hlm[8].y < hlm[6].y:
                    pose = "pointing"
                # Prabowo (Kepalan tangan)
                elif all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]):
                    pose = "video"

            self.last_pose = pose

        # RENDER OVERLAY (Picture-in-Picture)
        meme = None
        if self.last_pose == "video" and self.cap:
            ret, v_f = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.cap.read()
            if ret:
                meme = v_f
        elif self.last_pose in AVAILABLE_MEMES:
            meme = AVAILABLE_MEMES[self.last_pose]

        if meme is not None:
            # Ukuran overlay 35% dari layar
            mw, mh = int(w * 0.35), int(h * 0.35)
            meme_res = cv2.resize(meme, (mw, mh))
            # Tempel di pojok kanan bawah
            img[h-mh-10:h-10, w-mw-10:w-10] = meme_res
            cv2.rectangle(img, (w-mw-10, h-mh-10),
                          (w-10, h-10), (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# --- 4. DEPLOY ---
st.info("💡 **Tips**: Gunakan kamera depan. Coba tunjuk jari atau kepalkan tangan!")

webrtc_streamer(
    key="funny-filter",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FunnyVideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]},
                       {"urls": ["stun:stun1.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True,
)
