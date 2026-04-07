import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# --- 1. పర్ఫెక్ట్ ఇంపోర్ట్ మెకానిజం (ఎర్రర్స్ రాకుండా) ---
try:
    import cv2
    from ultralytics import YOLOWorld
except ImportError as e:
    st.error(f"లైబ్రరీ లోడింగ్ సమస్య: {e}")
    st.info("చిట్కా: మీ requirements.txt లో 'opencv-python-headless' ఉందో లేదో చూడండి.")
    st.stop()

# --- 2. పేజీ కాన్ఫిగరేషన్ ---
st.set_page_config(page_title="AI Onion Detector", page_icon="🧅", layout="centered")

# --- 3. AI మోడల్ లోడింగ్ (Caching ద్వారా వేగంగా) ---
@st.cache_resource
def load_onion_model():
    try:
        # 's' వెర్షన్ మొబైల్/క్లౌడ్ లో వేగంగా పనిచేస్తుంది
        model = YOLOWorld('yolov8s-world.pt')
        # కేవలం ఉల్లిపాయను మాత్రమే గుర్తించేలా సెట్ చేస్తున్నాం
        model.set_classes(["onion"])
        return model
    except Exception as ex:
        st.error(f"మోడల్ లోడ్ అవ్వలేదు: {ex}")
        return None

# మోడల్ ప్రారంభం
model = load_onion_model()

# --- 4. యూజర్ ఇంటర్ఫేస్ (UI) ---
st.title("🧅 Smart Onion Detector AI")
st.write("ఈ యాప్ కెమెరాను ఉపయోగించి ఉల్లిపాయను ఆటోమేటిక్‌గా గుర్తిస్తుంది.")

# కెమెరా ఇన్పుట్
img_file = st.camera_input("ఉల్లిపాయను కెమెరా ముందు ఉంచి ఫోటో తీయండి")

if img_file is not None:
    # ఫోటోను రీడ్ చేయడం
    input_img = Image.open(img_file)
    img_array = np.array(input_img)
    
    # AI తో డిటెక్షన్ రన్ చేయడం
    if model:
        with st.spinner('విశ్లేషిస్తున్నాను...'):
            # conf=0.3 అంటే 30% ఖచ్చితత్వం ఉంటే చూపిస్తుంది
            results = model.predict(img_array, conf=0.3)
            
            # రిజల్ట్ బాక్సులను డ్రా చేయడం
            res_plotted = results[0].plot()
            
            # OpenCV BGR వాడుతుంది, Streamlit కోసం RGB కి మార్చాలి
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # స్క్రీన్ మీద ఇమేజ్ చూపించడం
            st.image(res_rgb, caption="AI Detection Result", use_container_width=True)
            
            # ఫలితాల సందేశం
            detections = len(results[0].boxes)
            if detections > 0:
                st.success(f"✅ {detections} ఉల్లిపాయ(లు) గుర్తించబడ్డాయి!")
                st.balloons()
            else:
                st.warning("⚠️ ఉల్లిపాయ కనిపించడం లేదు. వెలుతురులో మళ్ళీ ప్రయత్నించండి.")
    else:
        st.error("AI మోడల్ అందుబాటులో లేదు.")

# అదనపు సమాచారం
st.divider()
st.caption("Developed for Venkat's Cooking Assistant Project")
