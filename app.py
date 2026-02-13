import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# Configuration
st.set_page_config(page_title="Email Spam Detector", page_icon="✉️")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spam_detector_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emails.csv')

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def main():
    st.title("✉️Email Spam Detector")
    st.markdown("Predict if an email is **Spam** or **Ham (Real)**")

    model_pipeline = load_model()

    if model_pipeline is None:
        st.error("⚠️ Model file not found! Please run your training script first.")
        return

    # --- DETECTION SECTION ---
    st.subheader("🔎 Analyze New Email")
    user_input = st.text_area("Paste the email content here:", placeholder="Type something...")

    if st.button("Analyze Email"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # Prediction
            prediction = model_pipeline.predict([user_input])[0]
        
            # Display Result
            if prediction == 1 == 'spam':
                st.error(f"🚨 This is likely SPAM email!")
            else:
                st.success(f"✅ This looks like a REAL email.")

    # --- DATASET ANALYTICS ---
    st.divider()
    if os.path.exists(DATA_PATH):
        st.subheader("📊 Training Data Insights")
        df = pd.read_csv(DATA_PATH)
        # Fix column naming for the chart
        fig = px.pie(df, names='Label', title="Distribution of Spam vs Ham", 
                     color='Label', color_discrete_map={'ham':'#2E7D32','spam':'#C62828'})
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()