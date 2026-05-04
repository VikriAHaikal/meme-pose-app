import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Import Mediapipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detector = mp_face.FaceMesh(
    refine_landmarks=True, min_detection_confidence=0.7)

# Load Assets (Pastikan file ada di GitHub)
MEMES = {
    "pointing": cv2.imread("assets/monkey_pointing.png"),
    "thinking": cv2.imread("assets/monkey_thinking.jpg"),
    "surprised": cv2.imread("assets/monkey_surprised.png"),
    "wink": cv2.imread("assets/monkey_wink.png")
}


class MemeProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture("assets/prabowo_video.mp4")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        # Deteksi Pose
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_hands = hands_detector.process(rgb)
        res_face = face_detector.process(rgb)

        pose = None
        is_prabowo = False

        # Logika Deteksi (Sederhana agar tidak berat)
        if res_face.multi_face_landmarks:
            flm = res_face.multi_face_landmarks[0].landmark
            if (flm[14].y - flm[13].y) > 0.05:
                pose = "surprised"

        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            if hlm[8].y < hlm[6].y:
                pose = "pointing"
            # Deteksi Prabowo
            is_fist = all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20])
            if is_fist and hlm[0].y < 0.4:
                is_prabowo = True
                pose = "prabowo"

        # Gabungkan Tampilan
        canvas = np.zeros((h * 2, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = img

        meme_area = np.zeros((h, w, 3), dtype=np.uint8)
        if is_prabowo:
            ret, v_f = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_f = self.video_cap.read()
            meme_area = cv2.resize(v_f, (w, h))
        elif pose in MEMES and MEMES[pose] is not None:
            meme_area = cv2.resize(MEMES[pose], (w, h))

        canvas[h:h*2, 0:w] = meme_area
        return frame.from_ndarray(canvas, format="bgr24")


st.title("🎭 AI Meme Pose Pro")

webrtc_streamer(
    key="meme-pro",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MemeProcessor,
    # Menambahkan server Google STUN agar koneksi lebih stabil
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
