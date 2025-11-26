import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# --- TITLE & HEADER ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 Plant Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Created by Aman Singh</h4>", unsafe_allow_html=True)
st.write("---")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Model wahi folder me hona chahiye jahan ye file hai
        model = tf.keras.models.load_model('trained_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- CLASS NAMES (38 Classes) ---
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot', 
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy', 
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight', 
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy', 
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy', 
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# --- UPLOAD SECTION ---
st.write("### 📸 Upload Leaf Image")
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None and model is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("🔍 Predict Disease"):
        with st.spinner('Analyzing...'):
            try:
                # Preprocessing
                img = image.resize((128, 128))
                img_array = np.array(img)
                
                # Convert grayscale to RGB if needed
                if img_array.shape[-1] != 3:
                     img_array = np.stack((img_array,)*3, axis=-1)
                     
                img_array = np.expand_dims(img_array, axis=0) # Batch dimension

                # Prediction
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Result Display
                st.success(f"*Prediction:* {predicted_class}")
                st.info(f"*Confidence:* {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")