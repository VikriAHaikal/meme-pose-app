import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PENANGANAN IMPORT MEDIAPIPE ---
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_mesh as mp_face
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.face_mesh as mp_face

# Inisialisasi Model AI
# refine_landmarks=True wajib untuk deteksi mata yang presisi (Wink)
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
face_detector = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.8)

# --- LOAD ASSETS ---
MEMES = {
    "pointing": cv2.imread("assets/monkey_pointing.png"),
    "thinking": cv2.imread("assets/monkey_thinking.jpg"), 
    "surprised": cv2.imread("assets/monkey_surprised.png"),
    "wink": cv2.imread("assets/monkey_wink.png")
}

class MemeExpertTransformer(VideoTransformerBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture("assets/prabowo_video.mp4")

    def get_ear(self, landmarks, pts):
        """Menghitung Eye Aspect Ratio (EAR) untuk deteksi mata tertutup"""
        p2_p6 = np.linalg.norm(np.array([landmarks[pts[1]].x, landmarks[pts[1]].y]) - 
                               np.array([landmarks[pts[5]].x, landmarks[pts[5]].y]))
        p3_p5 = np.linalg.norm(np.array([landmarks[pts[2]].x, landmarks[pts[2]].y]) - 
                               np.array([landmarks[pts[4]].x, landmarks[pts[4]].y]))
        p1_p4 = np.linalg.norm(np.array([landmarks[pts[0]].x, landmarks[pts[0]].y]) - 
                               np.array([landmarks[pts[3]].x, landmarks[pts[3]].y]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def transform(self, frame):
        # 1. Olah frame kamera (Sisi Kiri)
        img_cam = frame.to_ndarray(format="bgr24")
        img_cam = cv2.flip(img_cam, 1) # Mirror
        h, w, c = img_cam.shape
        rgb = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)

        # 2. Siapkan area meme (Sisi Kanan)
        meme_area = np.zeros((h, w, c), dtype=np.uint8)
        cv2.putText(meme_area, "MENUNGGU POSE...", (w//4, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        res_hands = hands_detector.process(rgb)
        res_face = face_detector.process(rgb)

        pose_detected = None
        is_prabowo = False

        # --- A. DETEKSI WAJAH (Wink & Surprised) ---
        if res_face.multi_face_landmarks:
            flm = res_face.multi_face_landmarks[0].landmark
            
            # Hitung EAR Mata Kiri & Kanan
            ear_l = self.get_ear(flm, [33, 160, 158, 133, 153, 144])
            ear_r = self.get_ear(flm, [362, 385, 387, 263, 373, 380])

            # Logika Wink: Jika satu mata jauh lebih tertutup dibanding mata lain
            if ear_l < (ear_r * 0.65): pose_detected = "wink"
            elif ear_r < (ear_l * 0.65): pose_detected = "wink"
            
            # Logika Surprised: Mulut terbuka
            if (flm[14].y - flm[13].y) > 0.055:
                pose_detected = "surprised"

        # --- B. DETEKSI TANGAN (Thinking, Pointing, Prabowo) ---
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            
            # Deteksi Thinking: Jari telunjuk (8) dekat ke Dagu (152)
            if res_face.multi_face_landmarks:
                dist_think = np.linalg.norm(np.array([hlm[8].x, hlm[8].y]) - 
                                           np.array([flm[152].x, flm[152].y]))
                if dist_think < 0.12: 
                    pose_detected = "thinking"

            # Deteksi Pointing (Hanya jika tidak sedang pose thinking)
            if pose_detected != "thinking":
                if hlm[8].y < hlm[6].y and hlm[12].y > hlm[10].y:
                    pose_detected = "pointing"

            # Deteksi Prabowo: Kepalan tangan diangkat tinggi
            is_fist = all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20])
            if is_fist and hlm[0].y < 0.4:
                is_prabowo = True
                pose_detected = "prabowo"

        # --- C. UPDATE STATE AUDIO ---
        st.session_state["play_audio"] = (pose_detected == "prabowo")

        # --- D. RENDERING SISI KANAN ---
        if is_prabowo:
            ret, v_frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, v_frame = self.video_cap.read()
            if ret:
                meme_area = cv2.resize(v_frame, (w, h))
        else:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if pose_detected in MEMES and MEMES[pose_detected] is not None:
                m_img = MEMES[pose_detected]
                # Beri border/margin kecil agar meme tidak menempel ke pinggir
                m_res = cv2.resize(m_img, (w-40, h-40))
                meme_area = np.zeros((h, w, c), dtype=np.uint8)
                meme_area[20:h-20, 20:w-20] = m_res

        # Gabungkan secara horizontal
        final_frame = np.hstack((img_cam, meme_area))
        # Garis pembatas putih di tengah
        cv2.line(final_frame, (w, 0), (w, h), (255, 255, 255), 2)
        
        return final_frame

# --- UI STREAMLIT ---
st.set_page_config(page_title="Meme AI Master", layout="wide")
st.title("🎭 AI Meme Pose Matcher (Split-Screen)")

if "play_audio" not in st.session_state:
    st.session_state["play_audio"] = False

# Trik Audio: Menggunakan st.audio yang autoplay saat pose Prabowo terdeteksi
if st.session_state["play_audio"]:
    st.audio("assets/prabowo_video.mp4", format="video/mp4", autoplay=True)

webrtc_streamer(
    key="meme-pro",
    video_transformer_factory=MemeExpertTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)