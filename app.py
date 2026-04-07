import streamlit as st
import numpy as np
from PIL import Image
import os

# --- 1. సర్వర్ గ్రాఫిక్స్ ఎర్రర్స్ రాకుండా సేఫ్టీ సెట్టింగ్ ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# లైబ్రరీలను ఇంపోర్ట్ చేయడం
try:
    import cv2
    from ultralytics import YOLOWorld
except ImportError as e:
    st.error(f"లైబ్రరీ లోడింగ్ సమస్య: {e}")
    st.info("చిట్కా: మీ packages.txt లో 'libgl1' మరియు 'libglib2.0-0' ఉన్నాయో లేదో చూడండి.")
    st.stop()

# --- 2. పేజీ కాన్ఫిగరేషన్ ---
st.set_page_config(page_title="Onion Detector AI", page_icon="🧅", layout="centered")

# --- 3. AI మోడల్ లోడింగ్ (Caching for Speed) ---
@st.cache_resource
def load_model():
    try:
        # తేలికపాటి వెర్షన్ (Small) మోడల్ వాడుతున్నాము
        model = YOLOWorld('yolov8s-world.pt')
        # కేవలం ఉల్లిపాయను (onion) మాత్రమే గుర్తించేలా సెట్ చేస్తున్నాం
        model.set_classes(["onion"])
        return model
    except Exception as e:
        st.error(f"మోడల్ లోడ్ అవ్వలేదు: {e}")
        return None

model = load_model()

# --- 4. యూజర్ ఇంటర్ఫేస్ (UI) ---
st.title("🧅 Smart Onion Detector AI")
st.write("ఈ అప్లికేషన్ మీ కెమెరాను ఉపయోగించి ఉల్లిపాయను గుర్తిస్తుంది.")

# కెమెరా ఇన్పుట్
img_file = st.camera_input("ఉల్లిపాయ ఫోటో తీయండి")

if img_file is not None:
    # 1. ఫోటోను రీడ్ చేయడం
    input_img = Image.open(img_file)
    img_array = np.array(input_img)
    
    # 2. AI డిటెక్షన్ రన్ చేయడం
    if model:
        with st.spinner('ఫోటోను విశ్లేషిస్తున్నాను...'):
            # conf=0.3 అంటే 30% ఖచ్చితత్వం ఉంటే చాలు
            results = model.predict(img_array, conf=0.3)
            
            # 3. రిజల్ట్ బాక్సులను ఫోటో మీద డ్రా చేయడం
            res_plotted = results[0].plot()
            
            # OpenCV BGR ఫార్మాట్ నుండి Streamlit RGB కి మార్పు (రంగులు మారకుండా ఉండటానికి)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # 4. స్క్రీన్ మీద రిజల్ట్ చూపించడం
            st.image(res_rgb, caption="AI విశ్లేషణ ఫలితం", use_container_width=True)
            
            # ఉల్లిపాయలు దొరికితే బెలూన్లు వస్తాయి
            if len(results[0].boxes) > 0:
                st.success(f"✅ {len(results[0].boxes)} ఉల్లిపాయ(లు) గుర్తించబడ్డాయి!")
                st.balloons()
            else:
                st.warning("⚠️ ఉల్లిపాయ ఏదీ కనిపించడం లేదు. ఫోటోని సరిగ్గా తీయండి.")
    else:
        st.error("AI మోడల్ అందుబాటులో లేదు.")

st.divider()
st.caption("Venkat's AI Project | Built with YOLO-World & Streamlit")
