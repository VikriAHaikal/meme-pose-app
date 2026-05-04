import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PROTEKSI IMPORT MEDIAPIPE (ANTI-CRASH) ---
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_mesh as mp_face
except:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.face_mesh as mp_face

# Inisialisasi Detektor (Sensitivitas Tinggi)
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
face_detector = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.8)

# --- LOAD ASSETS ---
# Menggunakan nama file yang Anda upload terakhir
MEMES = {
    "pointing": cv2.imread("assets/monkey_pointing.png"),
    "thinking": cv2.imread("assets/monkey_thinking.jpg"), 
    "surprised": cv2.imread("assets/monkey_surprised.png"),
    "wink": cv2.imread("assets/monkey_wink.png")
}

class MobileMemeExpert(VideoTransformerBase):
    def __init__(self):
        self.video_cap = cv2.VideoCapture("assets/prabowo_video.mp4")

    def get_ear(self, landmarks, pts):
        """Menghitung Eye Aspect Ratio (EAR) untuk deteksi kedip yang akurat"""
        p2_p6 = np.linalg.norm(np.array([landmarks[pts[1]].x, landmarks[pts[1]].y]) - 
                               np.array([landmarks[pts[5]].x, landmarks[pts[5]].y]))
        p3_p5 = np.linalg.norm(np.array([landmarks[pts[2]].x, landmarks[pts[2]].y]) - 
                               np.array([landmarks[pts[4]].x, landmarks[pts[4]].y]))
        p1_p4 = np.linalg.norm(np.array([landmarks[pts[0]].x, landmarks[pts[0]].y]) - 
                               np.array([landmarks[pts[3]].x, landmarks[pts[3]].y]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def transform(self, frame):
        img_cam = frame.to_ndarray(format="bgr24")
        img_cam = cv2.flip(img_cam, 1) # Efek Cermin
        h, w, c = img_cam.shape
        rgb = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)

        # Proses AI
        res_hands = hands_detector.process(rgb)
        res_face = face_detector.process(rgb)

        pose_detected = None
        is_prabowo = False

        # 1. LOGIKA WAJAH (Wink & Surprised)
        if res_face.multi_face_landmarks:
            flm = res_face.multi_face_landmarks[0].landmark
            ear_l = self.get_ear(flm, [33, 160, 158, 133, 153, 144])
            ear_r = self.get_ear(flm, [362, 385, 387, 263, 373, 380])

            # Perbandingan Relatif (Lebih Sensitif)
            if ear_l < (ear_r * 0.65) or ear_r < (ear_l * 0.65):
                pose_detected = "wink"
            
            if (flm[14].y - flm[13].y) > 0.055:
                pose_detected = "surprised"

        # 2. LOGIKA TANGAN (Thinking, Pointing, Prabowo)
        if res_hands.multi_hand_landmarks:
            hlm = res_hands.multi_hand_landmarks[0].landmark
            
            # Thinking: Ujung jari telunjuk (8) dekat ke dagu (152)
            if res_face.multi_face_landmarks:
                flm_ref = res_face.multi_face_landmarks[0].landmark
                dist_think = np.linalg.norm(np.array([hlm[8].x, hlm[8].y]) - 
                                           np.array([flm_ref[152].x, flm_ref[152].y]))
                if dist_think < 0.12: pose_detected = "thinking"

            # Pointing
            if pose_detected != "thinking":
                if hlm[8].y < hlm[6].y and hlm[12].y > hlm[10].y:
                    pose_detected = "pointing"

            # Prabowo: Kepalan tangan (fist) tinggi
            is_fist = all(hlm[i].y > hlm[i-2].y for i in [8, 12, 16, 20])
            if is_fist and hlm[0].y < 0.4:
                is_prabowo = True
                pose_detected = "prabowo"

        # --- SISTEM AUDIO STREAMLIT ---
        st.session_state["play_audio"] = is_prabowo

        # --- RENDERING UNTUK TAMPILAN HP (VERTICAL) ---
        # Kita buat kanvas yang memanjang ke bawah
        # Atas: Kamera (H), Bawah: Meme (H) -> Total Tinggi = 2H
        final_canvas = np.zeros((h * 2, w, 3), dtype=np.uint8)
        
        # Tempel Kamera di Atas
        final_canvas[0:h, 0:w] = img_cam
        
        # Siapkan Area Meme di Bawah
        meme_area = np.zeros((h, w, 3), dtype=np.uint8)
        
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
                # Resize meme agar pas di kotak bawah
                m_img = MEMES[pose_detected]
                m_res = cv2.resize(m_img, (w-20, h-20))
                meme_area[10:h-10, 10:w-10] = m_res
            else:
                cv2.putText(meme_area, "MENUNGGU POSE...", (w//5, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Tempel Meme di Bawah
        final_canvas[h:h*2, 0:w] = meme_area
        
        # Garis Pembatas
        cv2.line(final_canvas, (0, h), (w, h), (255, 255, 255), 3)
        
        return final_canvas

# --- UI STREAMLIT ---
st.set_page_config(page_title="Meme AI Pro", layout="centered")

# CSS Khusus HP agar responsif
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-align: center; font-size: 24px; }
    video { width: 100% !important; border-radius: 15px; border: 2px solid #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 AI Meme Pose Expert")

# Logika Audio Otomatis
if "play_audio" not in st.session_state:
    st.session_state["play_audio"] = False

if st.session_state["play_audio"]:
    st.audio("assets/prabowo_video.mp4", format="video/mp4", autoplay=True)

webrtc_streamer(
    key="mobile-meme",
    video_transformer_factory=MobileMemeExpert,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.info("💡 Tips: Gunakan mode Portrait di HP. Tunjuk layar (👉), Mikir (🤔), Kedip (😉), atau Kepal Tangan (✊)!")