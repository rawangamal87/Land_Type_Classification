import streamlit as st
import pandas as pd
import pickle
from land_type_classification import preprocess_data  

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Land Type Classification", layout="centered")
st.title("🌍 Land Type Classification App")
st.write("Upload a CSV file to classify land types using a pre-trained ML model.")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("⚙️ Processing & Predicting...")
        processed_df = preprocess_data(df)
        predictions = model.predict(processed_df)

        st.subheader("🧠 Predictions")
        df["Prediction"] = predictions
        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {e}")
