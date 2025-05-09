import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Land Type Classification", layout="centered")
st.title("🌍 Land Type Classification using Sentinel-2")

st.markdown("Upload a satellite image (Sentinel-2)، The trained model will classify the land type..")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("🔼Upload an image", type=["jpg", "png", "tif"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="the uploaded image", use_column_width=True)

    st.write("🛠️ Processing...")
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    band_names = [
        'B01 - Coastal Aerosol','B02 - Blue','B03 - Green','B04 - Red','B05 - Red Edge 1','B06 - Red Edge 2','B07 - Red Edge 3','B08 - NIR',
        'B8A - Narrow NIR','B09 - Water Vapor','B10 - SWIR - Cirrus','B11 - SWIR 1','B12 - SWIR 2'
    ]

    st.success(f"🌟 prediction: {band_names[predicted_class]}")