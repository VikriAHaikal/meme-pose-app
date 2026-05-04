import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Import Mediapipe dengan proteksi
try:
    import mediapipe as mp
except ImportError:
    import mediapipe.python.solutions as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detector = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

# Load Assets
MEMES = {
    "pointing": cv2.imread("assets/monkey_pointing.png"),
    "thinking": cv2.imread("assets/monkey_thinking.jpg"), 
    "surprised": cv2.imread("assets/monkey_surprised.png"),
    "wink": cv2.imread("assets/monkey_wink.png")
}

class MemeProcessor(VideoProcessorBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture("assets/prabowo_video.mp4")

    def get_ear(self, landmarks, pts):
        p2_p6 = np.linalg.norm(np.array([landmarks[pts[1]].x, landmarks[pts[1]].y]) - 
                               np.array([landmarks[pts[5]].x, landmarks[pts[5]].y]))
        p3_p5 = np.linalg.norm(np.array([landmarks[pts[2]].x, landmarks[pts[2]].y]) - 
                               np.array([landmarks[pts[4]].x, landmarks[pts[4]].y]))
        p1_p4 = np.linalg.norm(np.array([landmarks[pts[0]].x, landmarks[pts[0]].y]) - 
                               np.array([landmarks[pts[3]].x, landmarks[pts[3]].y]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res_hands = hands_detector.process(rgb)
        res_face = face_detector.process(rgb)

        pose = None
        is_prabowo = False

        if res_face.multi_face_landmarks:
            flm = res_face.multi_face_landmarks[0].landmark
            ear_l = self.get_ear(flm, [33, 160, 158, 133, 153, 144])
            ear_r = self.get_ear(flm, [362, 385, 387, 263, 373, 380])
            if ear_l < (ear_r * 0.65) or ear_r < (ear_l * 0.65): pose = "wink"
            if (flm[14].y - flm[13].y) > 0.055: pose = "surprised"

        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            if res_face.multi_face_landmarks:
                dist = np.linalg.norm(np.array([hlm[8].x, hlm[8].y]) - 
                                     np.array([flm[152].x, flm[152].y]))
                if dist < 0.12: pose = "thinking"
            
            if pose != "thinking":
                if hlm[8].y < hlm[6].y and hlm[12].y > hlm[10].y: pose = "pointing"

            is_fist = all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20])
            if is_fist and hlm[0].y < 0.4:
                is_prabowo = True
                pose = "prabowo"

        # Gabungkan Tampilan (Vertical untuk HP)
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
            m_res = cv2.resize(MEMES[pose], (w-20, h-20))
            meme_area[10:h-10, 10:w-10] = m_res

        canvas[h:h*2, 0:w] = meme_area
        return frame.from_ndarray(canvas, format="bgr24")

st.title("🎭 AI Meme Pose Pro")
webrtc_streamer(
    key="meme-app",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MemeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)