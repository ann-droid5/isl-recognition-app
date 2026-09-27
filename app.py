import streamlit as st
import os
import tempfile
from video_prediction import load_model_and_classes, predict_video, get_final_prediction

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="ISL Recognition",
    
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .result-card {
        background-color: #1e1e1e;
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .result-title {
        color: #a0a0a0;
        font-size: 20px;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-word {
        color: #4CAF50;
        font-size: 64px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 3px;
        text-transform: capitalize;
    }
    /* Style adaptation for light mode */
    @media (prefers-color-scheme: light) {
        .result-card { background-color: #f8f9fa; border: 2px solid #28a745; }
        .result-title { color: #555; }
        .result-word { color: #28a745; }
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER ---
st.markdown("<h1 class='main-header'> Indian Sign Language Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='main-header'>Video-Based ISL Word Recognition and Text Conversion</h4>", unsafe_allow_html=True)
st.write("Upload a video containing an Indian Sign Language sign and our AI model will recognize the sign and convert it into text.")
st.divider()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("About the Project")
    st.info(
        "This application uses a MobileNetV2 deep-learning model trained on an Indian Sign Language image dataset. "
        "Video frames are analyzed using OpenCV and individual predictions are combined to identify the final sign."
    )
    
    st.subheader("Technology Stack")
    st.markdown("""
    - **Model:** MobileNetV2
    - **Framework:** TensorFlow / Keras
    - **Video Processing:** OpenCV
    - **Classes:** 110 ISL Signs
    
    """)
    
    st.subheader("How it works")
    st.markdown("""
    1. 🎥 Upload video
    2. 🎞️ Extract video frames
    3. 🤖 AI predicts each frame
    4. 📊 Combine predictions
    5. ✨ Display recognized sign
    """)
    
    

# --- 4. CONSTANTS & CACHING ---
@st.cache_resource(show_spinner=False)
def load_ai_model():
    """
    Caches the model and class names so it only loads once on startup.
    This prevents the app from being slow when processing multiple videos.
    """
    return load_model_and_classes()

# --- 5. HELPER FUNCTIONS ---
def format_class_name(name):
    """Formats raw class names (e.g., 'All_Gone' -> 'All Gone', 'I_m Good' -> 'I'm Good')."""
    name = str(name)
    name = name.replace("_", " ")
    name = name.replace("I m", "I'm")
    return name.title() # Capitalizes the first letter of each word (e.g., "Mother")

# --- 6. MODEL INITIALIZATION ---
try:
    model, class_names = load_ai_model()
except Exception as e:
    st.error(f"Failed to load model or classes: {str(e)}")
    st.stop()  # Stop execution until files are placed in the directory

# --- 7. MAIN VIDEO UPLOAD AREA ---
uploaded_file = st.file_uploader(
    "Upload an ISL Video", 
    type=["mp4", "avi", "mov"], 
    help="Upload a short video showing one ISL sign."
)

if uploaded_file is not None:
    # Display basic video information
    st.write(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / (1024*1024):.2f} MB")
    
    # Display video preview
    st.video(uploaded_file)
    
    # Ensure prediction is user-triggered, not automatic
    if st.button(" Recognize Sign", type="primary", use_container_width=True):
        
        # --- 8. PROCESSING SECTION ---
        with st.spinner("Analyzing video frames... Please wait."):
            
            # Temporarily save the uploaded video to disk so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            try:
                # Call YOUR video prediction functions
                predictions = predict_video(
                    video_path=temp_video_path,
                    model=model,
                    class_names=class_names
                )
                
                result = get_final_prediction(predictions)
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                result = None
            finally:
                # Clean up the temporary file immediately after processing
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # --- 9. RESULT SECTION ---
        if result:
            if result == "No confident prediction":
                st.warning("No confident sign was detected. Please upload a clearer video.")
            else:
                formatted_class = format_class_name(result)
                
                # Render the prominent result card
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">Recognized Sign</div>
                        <div class="result-word">{formatted_class}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- 10. RESET / NEW VIDEO ---
                st.write("")
                if st.button(" Try Another Video", use_container_width=True):
                    # Rerunning Streamlit clears the current UI state to accept a new video seamlessly
                    st.rerun()
