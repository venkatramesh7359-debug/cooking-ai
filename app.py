import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

# Import YOLO after other basic imports to avoid conflict
try:
    from ultralytics import YOLOWorld
except ImportError:
    st.error("Ultralytics library missing. Please check requirements.txt")

# --- Page Config ---
st.set_page_config(page_title="AI Ingredient Detector", page_icon="🧅")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_onion_model():
    # Using 's' (small) version for better balance between speed and accuracy
    model = YOLOWorld('yolov8s-world.pt')
    # Set the model to ONLY detect onions
    model.set_classes(["onion"])
    return model

# Initialize model
model = load_onion_model()

# --- UI Setup ---
st.title("🧅 Onion Detector AI")
st.write("ఈ అప్లికేషన్ మీ కెమెరాను ఉపయోగించి ఉల్లిపాయను గుర్తిస్తుంది.")

# Camera input
img_file = st.camera_input("కెమెరా ముందు ఉల్లిపాయను ఉంచి ఫోటో తీయండి")

if img_file is not None:
    # 1. Convert to PIL Image
    input_img = Image.open(img_file)
    
    # 2. Convert to Numpy array for YOLO
    img_array = np.array(input_img)
    
    # 3. Run Detection
    with st.spinner('Analyzing image...'):
        # conf=0.3 means 30% confidence threshold
        results = model.predict(img_array, conf=0.3)
        
        # 4. Visualization
        # Plotting boxes on the image
        res_plotted = results[0].plot()
        
        # Convert BGR (OpenCV default) to RGB for Streamlit display
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 5. Display Result
        st.image(res_rgb, caption="Processed Image", use_container_width=True)
        
        # Check if any onions were detected
        if len(results[0].boxes) > 0:
            st.success(f"✅ ఉల్లిపాయను గుర్తించాను! ({len(results[0].boxes)} found)")
            st.balloons() # Victory animation!
        else:
            st.warning("⚠️ ఉల్లిపాయ కనిపించడం లేదు. మళ్ళీ ప్రయత్నించండి.")

st.divider()
st.info("గమనిక: మొదటిసారి రన్ అయ్యేటప్పుడు AI మోడల్ డౌన్‌లోడ్ అవ్వడానికి 1-2 నిమిషాలు పడుతుంది. ఓపిక పట్టండి.")
