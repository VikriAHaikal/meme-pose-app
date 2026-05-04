import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. UI MOBILE-OPTIMIZED ---
st.set_page_config(page_title="Pose Camera Detection", layout="centered")
st.markdown("""
    <style>
    .stVideo, video { width: 100% !important; border-radius: 15px; border: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("Pose Camera Detection")

# --- 2. KONFIGURASI ASSETS (Check & Load) ---
ASSETS_PATH = "assets"
MEME_FILES = {
    "pointing": "monkey_pointing.png",
    "thinking": "monkey_thinking.png",
    "surprised": "monkey_surprised.png",
    "wink": "monkey_wink.png"
}
VIDEO_FILE = "prabowo_video.mp4"


@st.cache_resource
def load_all_assets():
    memes = {}
    for key, name in MEME_FILES.items():
        p = os.path.join(ASSETS_PATH, name)
        if not os.path.exists(p):
            p = p.replace(".png", ".jpg")  # Cek fallback jpg
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                memes[key] = img

    # Detector di-cache agar tidak berat
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    h_det = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)
    f_det = mp_face.FaceMesh(refine_landmarks=False,
                             min_detection_confidence=0.5)
    return memes, h_det, f_det


AVAILABLE_MEMES, HANDS_DET, FACE_DET = load_all_assets()
VIDEO_PATH = os.path.join(ASSETS_PATH, VIDEO_FILE)

# --- 3. LOGIKA TURBO PROCESSOR ---


class TurboOverlayProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture(
            VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None
        self.frame_skip = 0
        self.last_pose = None

    def get_ear(self, landmarks, pts):
        p1, p2, p3, p4, p5, p6 = [
            np.array([landmarks[i].x, landmarks[i].y]) for i in pts]
        return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # --- FRAME SKIPPING (Anti Slowmo) ---
        self.frame_skip += 1
        if self.frame_skip % 3 == 0:  # Cuma proses AI tiap 3 frame
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_hands = HANDS_DET.process(rgb)
            res_face = FACE_DET.process(rgb)

            new_pose = None
            # Logika Wajah
            if res_face.multi_face_landmarks:
                flm = res_face.multi_face_landmarks[0].landmark
                if (flm[14].y - flm[13].y) > 0.06:
                    new_pose = "surprised"
                else:
                    el = self.get_ear(flm, [33, 160, 158, 133, 153, 144])
                    er = self.get_ear(flm, [362, 385, 387, 263, 373, 380])
                    if el < (er * 0.6) or er < (el * 0.6):
                        new_pose = "wink"

            # Logika Tangan
            if res_hands.multi_hand_landmarks:
                hlm = res_hands.multi_hand_landmarks[0].landmark
                # Thinking (Tangan di Dagu)
                if res_face.multi_face_landmarks:
                    chin = res_face.multi_face_landmarks[0].landmark[152]
                    dist = np.linalg.norm(
                        np.array([hlm[8].x, hlm[8].y]) - np.array([chin.x, chin.y]))
                    if dist < 0.15:
                        new_pose = "thinking"

                if not new_pose:
                    if hlm[8].y < hlm[6].y:
                        new_pose = "pointing"
                    elif all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]):
                        new_pose = "video"

            self.last_pose = new_pose

        # --- RENDER OVERLAY (PIP) ---
        meme_img = None
        if self.last_pose == "video" and self.video_cap:
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            if ret:
                meme_img = v_f
        elif self.last_pose in AVAILABLE_MEMES:
            meme_img = AVAILABLE_MEMES[self.last_pose]

        if meme_img is not None:
            pw, ph = int(w * 0.35), int(h * 0.35)
            meme_res = cv2.resize(meme_img, (pw, ph))
            img[h-ph-10:h-10, w-pw-10:w-10] = meme_res
            cv2.rectangle(img, (w-pw-10, h-ph-10),
                          (w-10, h-10), (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# --- 4. START STREAMER ---
st.info("Pose: Tunjuk, Mikir (Tangan di dagu), Kedip, Kaget (Mangap), atau Kepal tangan.")

webrtc_streamer(
    key="meme-pro-v4",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=TurboOverlayProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            # Tambahan agar lebih stabil
            {"urls": ["stun:stun.services.mozilla.com"]}
        ]
    },
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True,
)
