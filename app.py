import streamlit as st
from ultralytics import YOLOWorld
import cv2
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Onion Detector AI", page_icon="🧅")

# 2. Load Model with Caching (ఎర్రర్స్ రాకుండా ఉండటానికి)
@st.cache_resource
def get_model():
    # YOLOWorld model can detect custom objects like 'onion'
    try:
        model = YOLOWorld('yolov8s-world.pt')
        model.set_classes(["onion"])
        return model
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

model = get_model()

# 3. User Interface
st.title("🧅 Smart Onion Detector")
st.write("కెమెరాను ఉపయోగించి ఉల్లిపాయను గుర్తించండి.")

# Camera Input
img_file = st.camera_input("Take a photo of an onion")

if img_file is not None:
    # Convert image for processing
    input_img = Image.open(img_file)
    img_array = np.array(input_img)
    
    # Run AI Detection
    if model:
        with st.spinner('Analyzing...'):
            results = model.predict(img_array, conf=0.3)
            
            # Draw results on the image
            res_plotted = results[0].plot()
            
            # Display results
            st.image(res_plotted, caption="Detection Result", use_column_width=True)
            
            # Check if onion found
            if len(results[0].boxes) > 0:
                st.success(f"✅ ఉల్లిపాయను గుర్తించాను! (Found {len(results[0].boxes)} onion/s)")
            else:
                st.warning("⚠️ ఉల్లిపాయ కనిపించడం లేదు. దయచేసి వెలుతురులో మళ్ళీ ప్రయత్నించండి.")
    else:
        st.error("Model could not be initialized.")

st.info("గమనిక: మొదటిసారి యాప్ ఓపెన్ అయినప్పుడు AI మోడల్ డౌన్‌లోడ్ అవ్వడానికి 1-2 నిమిషాలు పడుతుంది.")
