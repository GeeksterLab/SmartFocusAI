# app.py

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, WebRtcStreamerState
from matplotlib.backends.backend_pdf import PdfPages

import av
import io
import time
import os 
import cv2

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from src.detection.typing_activity import TypingActivityDetector
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector
from src.gaze.combined_gaze import CombinedGazeDetector
from src.gaze.smoother import DirectionSmoother
from src.logic.focus_manager import FocusManager


st.set_page_config(page_title="Smart Focus AI", layout="wide")
st.title("🎯 :rainbow[SMART FOCUS AI]")
st.warning("Don't forget to allow webcam access in your browser settings.")
st.markdown("""
    <a href="#live">
        <button style='font-size:16px;padding:10px 20px;margin-top:20px;'>🚀 Try the real-time detection now by clicking START</button>
    </a>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    video {
        max-width: 150px !important;
        height: auto !important;
        margin: 10px auto;
        display: block;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    video:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }

    footer { visibility: hidden !important; }

    .footer {
    position: fixed !important;;
    left: 0 !important;;
    bottom: 0 !important;;
    width: 100% !important;;
    text-align: right !important;;        
    padding: 8px 16px !important;;       
    background-color: rgba(255,255,255,0.8) !important;;
    color: #666 !important;;
    font-size: 0.8rem !important;;
    z-index: 9999 !important;;           
        }
    </style>
""", unsafe_allow_html=True)

# -- Logging parameters --
LOG_DIR = "Logs"
os.makedirs(LOG_DIR, exist_ok=True)
today = pd.Timestamp.now().date()
FILENAME_CSV = os.path.join(LOG_DIR, f"focus_logs_{today}.csv")

# Initialize the DataFrame in session
if "df_log" not in st.session_state:
    st.session_state.df_log = pd.DataFrame(columns=["timestamp","focused", "typing","gaze","blink_count"])
if "recording" not in st.session_state:
    st.session_state.recording = False


def make_pdf(df: pd.DataFrame) -> bytes:
    # 1) Timestamp en datetime
    if df["timestamp"].dtype != "datetime64[ns]":
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # 2) Tri et calcul des intervalles dt (en secondes)
    df = df.sort_values("timestamp")
    df["dt"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    # 3) Durées cumulées
    total_secs      = df["dt"].sum()
    focus_secs      = df.loc[df.focused == 1, "dt"].sum()
    distracted_secs = df.loc[df.focused == 0, "dt"].sum()
    typing_secs     = df.loc[df.typing  == 1, "dt"].sum()
    gaze_secs       = df.loc[df.gaze    == 1, "dt"].sum()
    blinks          = int(df["blink_count"].max())

    # 4) Taux time-weighted [%]
    focus_rate      = (focus_secs      / total_secs * 100) if total_secs else 0
    distracted_rate = (distracted_secs / total_secs * 100) if total_secs else 0
    typing_rate     = (typing_secs     / total_secs * 100) if total_secs else 0
    gaze_rate       = (gaze_secs       / total_secs * 100) if total_secs else 0

    # 5) Génération du PDF
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- Page 1 : résumé des stats ---
        fig, ax = plt.subplots(figsize=(6,4))
        stats_text = (
            f"Focus rate:      {focus_rate:.1f}%\n"
            f"Distracted rate: {distracted_rate:.1f}%\n"
            f"Typing rate:     {typing_rate:.1f}%\n"
            f"Gaze rate:       {gaze_rate:.1f}%\n"
            f"Blinks:          {blinks}"
        )
        ax.text(0.1, 0.5, stats_text, fontsize=14, family="monospace")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2 : évolution dans le temps ---
        fig, ax = plt.subplots(figsize=(8,4))
        df_plot = df.set_index("timestamp")[["focused","typing","gaze"]]
        df_plot.rolling(30).mean().plot(ax=ax)
        ax.set_ylabel("Rolling average")
        ax.set_title("Focus / Typing / Gaze over time")
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()



class FocusTransformer(VideoProcessorBase):
    def __init__(self):
        self.typing = TypingActivityDetector(display_timeout=0.5)
        self.typing.start()
        self.face_mesh = FaceMeshDetector()
        self.blink = BlinkDetector()
        self.gaze = CombinedGazeDetector(self.face_mesh)
        self.focus_mgr = FocusManager(typing_detector=self.typing, gaze_detector=self.gaze)
        self.h_smoother = DirectionSmoother(window_size=5)
        self.v_smoother = DirectionSmoother(window_size=5)
        # Cycle duration (visible + hidden) in seconds
        self.cycle = 1.0  

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        lm = self.face_mesh.process(img)
        if lm is not None:
            self.blink.update(img, lm, self.face_mesh)
        is_typing = self.typing.is_typing()
        is_gazing = self.gaze.is_gazing(img)
        focused = self.focus_mgr.is_focused(img)
        ts = time.time()
        new_row = {
            "timestamp": ts,
            "focused": int(focused),
            "typing": int(is_typing),
            "gaze": int(is_gazing),
            "blink_count": self.blink.blink_count
        }
        pd.DataFrame([new_row]).to_csv(FILENAME_CSV, mode="a", header=False, index=False)
        status = "Focused" if focused else "Distracted"
        color = (0,255,0) if focused else (0,0,255)
        cv2.putText(img, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Typing:{int(is_typing)}  Center Gaze:{int(is_gazing)}  Blink:{self.blink.blink_count}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Since OpenCV can't align text to the right by itself -->
        # get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            "HELLO STREAMLIT",
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2
        )

        # Compute phase between 0 and 1
        phase = (time.time() % self.cycle) / self.cycle

        # If phase < 0.5: draw text, otherwise don't → blinking
        if phase < 0.5:
            text = "HELLO STREAMLIT"
            (w, h), _ = cv2.getTextSize(text,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,
                                        thickness=2)
            margin = 10
            x = img.shape[1] - w - margin
            y = h + margin

            cv2.putText(img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
st.markdown("## 🧠 :rainbow[What is Smart Focus AI]")
st.markdown("""

**Type** : Interactive Web Application   
**Technologies** : Python · Streamlit · OpenCV · MediaPipe · Light Machine Learning

---

**Description :**  
Smart Focus AI is an intelligent web application that analyzes the user’s **attention** in real time via the **webcam**. It combines several detectors to continuously assess the user’s focus state:
- ✅ Keyboard typing detection (*typing*)
- 🎯 Gaze tracking (*center gaze / gaze away*)
- 👁️ Blink detection (*blinks*)
- 🔄 Signal fusion to determine whether the user is **Focused** or **Distracted**

The interface provides:

- 📽️ Live visual feedback (video overlay) in the **left sidebar**
- 📊 Automatic reporting in **PDF** and **CSV** formats via the **left sidebar**
- 🎓 Built-in explanatory videos to guide the user
""")

st.markdown("## 🎓 :rainbow[How to Use Smart Focus AI]")

with st.expander("🎓 How does Smart Focus AI work?"):
    st.markdown("""
    1. The application activates your webcam (with your permission).
    2. It continuously monitors:
       - 🤓 your gaze direction (center or away),
       - 🖐️ your typing activity,
       - 👁️ your blinking frequency.
    3. These signals are combined to determine whether you're **Focused** or **Distracted**.
    4. At the end of the session, you get a **clear and exportable report** (CSV + PDF).
    """)

st.info("Tutorial : Here are 3 short videos (5s) to help you understand how the system works:")
st.write(" - 📽️ These videos illustrate the behaviors the system detects (not live recording).")

st.markdown('<div id="live"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✍️ Typing Detection", "🎯 Center Gaze", "🙈 Gaze Away"])

with tab1:
    st.markdown("**Typing Detection:** When you type, the system tracks activity. You will see -> Focused, Typing:1, Center Gaze:1")
    st.video("static/tutorials/only_typing.mp4", format="video/mp4", start_time=0)

with tab2:
    st.markdown("**Center Gaze:** When you look at the screen, you're marked as focused. You will see -> Focused, Typing: 0, Center Gaze:1")
    st.video("static/tutorials/only_gaze_center.mp4", format="video/mp4", start_time=0)

with tab3:
    st.markdown("**Gaze Away:** When you look away from the screen, you're marked as distracted. You will see --> Distracted, Typing: 0, Center Gaze:0")
    st.video("static/tutorials/gaze_away.mp4", format="video/mp4", start_time=0)


webrtc_ctx = webrtc_streamer(
    key="smart-focus",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FocusTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Managing webcam state to detect stop ---
if webrtc_ctx:
    prev_state = st.session_state.get("prev_webrtc_state", None)
    current_state = webrtc_ctx.state

    if prev_state == "PLAYING" and current_state != "PLAYING":
        st.session_state["show_success"] = True
        st.session_state["success_start_time"] = time.time()

    st.session_state["prev_webrtc_state"] = current_state

# --- Display while webcam is running ---
if webrtc_ctx and webrtc_ctx.state.playing:
    st.markdown("🔴 **Recording in progress...**", unsafe_allow_html=True)
    live_chart = st.empty()

    for _ in range(100):
        if not webrtc_ctx.state.playing:
            break
        df_live = st.session_state.df_log.copy()

        if not df_live.empty:
            if not np.issubdtype(df_live["timestamp"].dtype, np.datetime64):
                df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], unit="s")

            now = pd.Timestamp.now()
            df_recent = df_live[df_live["timestamp"] > now - pd.Timedelta(seconds=120)]
            df_recent.set_index("timestamp", inplace=True)

            live_chart.line_chart(df_recent[["focused", "typing", "gaze"]])
        else:
            live_chart.info("⏳ Waiting for focus data...")
        time.sleep(0.5)

# --- Display after webcam stops ---
if st.session_state.get("show_success", False):
    elapsed = time.time() - st.session_state["success_start_time"]
    if elapsed < 10:
        st.success("✅ Analysis completed successfully")
    else:
        st.session_state["show_success"] = False

# --- Sidebar ---
st.sidebar.markdown("<h1 style='text-align: center;'>Home</h1>", unsafe_allow_html=True)



def format_duration(seconds: float) -> str:
    """Formats a duration in seconds as 'Xh Ym Zs' or 'Ym Zs'."""
    hours, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"

if os.path.exists(FILENAME_CSV):
    with st.spinner("📊 Analyzing data..."):
        df_today = pd.read_csv(
            FILENAME_CSV,
            header=None,
            names=["timestamp", "focused", "typing", "gaze", "blink_count"]
        )
    # 1) Convert timestamp
    df_today["timestamp"] = pd.to_datetime(df_today["timestamp"], unit="s")
    # 2) Sort chronologically + compute intervals (dt)
    df_today = df_today.sort_values("timestamp")
    df_today["dt"] = df_today["timestamp"].diff().dt.total_seconds().fillna(0)
    # 3) Total duration
    total_secs = (df_today["timestamp"].iloc[-1] - df_today["timestamp"].iloc[0]).total_seconds()
    total_fmt  = format_duration(total_secs)

    # 4) Effective durations
    focus_secs      = df_today.loc[df_today.focused == 1, "dt"].sum()
    distracted_secs = df_today.loc[df_today.focused == 0, "dt"].sum()
    typing_secs     = df_today.loc[df_today.typing  == 1, "dt"].sum()

    # 5) Time-based rates [%]
    focus_rate      = focus_secs      / total_secs * 100 if total_secs else 0
    distracted_rate = distracted_secs / total_secs * 100 if total_secs else 0
    typing_rate     = typing_secs     / total_secs * 100 if total_secs else 0

    # 6) Format durations for display
    focus_fmt      = format_duration(focus_secs)
    distracted_fmt = format_duration(distracted_secs)
    typing_fmt     = format_duration(typing_secs)

    # --- Build sidebar ---
    st.sidebar.header("📊 Report of the Day", divider="rainbow")
    st.sidebar.markdown(
        "<p style='text-align: center;'>Here is today's summary of your focus analysis.</p>",
        unsafe_allow_html=True
    )

    # Focus
    st.sidebar.subheader("Focus", divider="blue")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      🧠 <strong>{focus_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{focus_fmt} focused over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Distracted
    st.sidebar.subheader("Distracted", divider="orange")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      👀 <strong>{distracted_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{distracted_fmt} distracted over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Typing
    st.sidebar.subheader("Typing", divider="violet")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      ✍️ <strong>{typing_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{typing_fmt} typed over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    csv_bytes = open(FILENAME_CSV, "rb").read()
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button("⬇️ Download the CSV", data=csv_bytes, file_name=os.path.basename(FILENAME_CSV), mime="text/csv")
        pdf_bytes = make_pdf(df_today)
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button("⬇️ Download the PDF", data=pdf_bytes, file_name=f"summary_{today}.pdf", mime="application/pdf")

        st.sidebar.markdown("<br>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ How it works?"):
    st.markdown("""
    - This app analyzes your attention using your webcam.
    - It detects typing, gaze, blinks and focus.
    - Click **Start** to begin, then **Stop** to generate the report.
    """)

# st.caption("<br>", unsafe_allow_html=True)
# st.caption("© 2025 GeeksterLab")

st.markdown(
    '<div class="footer">© 2025 GeeksterLab</div>',
    unsafe_allow_html=True
)
