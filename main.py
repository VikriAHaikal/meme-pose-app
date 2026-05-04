import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import mediapipe as mp

# --- 1. SETTING UI MOBILE FIRST ---
st.set_page_config(page_title="Pose Camera Detection", layout="centered")

st.markdown("""
    <style>
    .stVideo, video {
        width: 100% !important;
        border-radius: 15px;
        border: 2px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Pose Camera Detection")

# --- 2. KONFIGURASI ASSETS ---
ASSETS_PATH = "assets"
MEME_FILES = {
    "pointing": "monkey_pointing.png",
    # Sesuaikan ekstensi (.png/.jpg) di GitHub kamu
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
        else:
            # Coba cari versi .jpg jika .png tidak ada
            p_alt = p.replace(".png", ".jpg")
            if os.path.exists(p_alt):
                img = cv2.imread(p_alt)
                if img is not None:
                    loaded[key] = img
    return loaded


AVAILABLE_MEMES = load_memes()
VIDEO_PATH = os.path.join(ASSETS_PATH, VIDEO_FILE)

# --- 3. LOGIKA AI (MEDIAPIPE) ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detector = mp_face.FaceMesh(
    refine_landmarks=True, min_detection_confidence=0.7)


class MobileProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture(
            VIDEO_PATH) if os.path.exists(VIDEO_PATH) else None

    def get_ear(self, landmarks, pts):
        # Hitung Eye Aspect Ratio (EAR) untuk deteksi kedip
        p1, p2, p3, p4, p5, p6 = [
            np.array([landmarks[i].x, landmarks[i].y]) for i in pts]
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return (v1 + v2) / (2.0 * h)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res_hands = hands_detector.process(rgb)
        res_face = face_detector.process(rgb)

        pose = None

        # A. Deteksi Wajah (Wink & Surprised)
        if res_face.multi_face_landmarks:
            flm = res_face.multi_face_landmarks[0].landmark
            # 1. Surprised (Mulut mangap)
            if (flm[14].y - flm[13].y) > 0.05:
                pose = "surprised"
            # 2. Wink (Satu mata tertutup)
            else:
                ear_l = self.get_ear(flm, [33, 160, 158, 133, 153, 144])
                ear_r = self.get_ear(flm, [362, 385, 387, 263, 373, 380])
                if ear_l < (ear_r * 0.6) or ear_r < (ear_l * 0.6):
                    pose = "wink"

        # B. Deteksi Tangan (Thinking, Pointing, Prabowo)
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            # 3. Thinking (Tangan dekat dagu)
            if res_face.multi_face_landmarks:
                chin = res_face.multi_face_landmarks[0].landmark[152]
                dist = np.linalg.norm(
                    np.array([hlm[8].x, hlm[8].y]) - np.array([chin.x, chin.y]))
                if dist < 0.15:
                    pose = "thinking"

            if pose is None:
                # 4. Pointing (Telunjuk naik)
                if hlm[8].y < hlm[6].y and hlm[12].y > hlm[10].y:
                    pose = "pointing"
                # 5. Prabowo (Kepalan tangan)
                elif all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20]):
                    pose = "video"

        # --- DRAW OVERLAY (PIP) ---
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
            # Overlay di pojok kanan bawah (35% dari ukuran layar)
            pw, ph = int(w * 0.35), int(h * 0.35)
            meme_res = cv2.resize(meme_img, (pw, ph))
            img[h-ph-10:h-10, w-pw-10:w-10] = meme_res
            cv2.rectangle(img, (w-pw-10, h-ph-10),
                          (w-10, h-10), (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")


# --- 4. RUNNER ---
st.info("💡 **Pose**: Tunjuk jari ke atas, Tangan di dagu (Mikir), Kedip satu mata, Kaget (Mangap), atau Kepal tangan.")

webrtc_streamer(
    key="mobile-final-pro",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MobileProcessor,
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True,
)
