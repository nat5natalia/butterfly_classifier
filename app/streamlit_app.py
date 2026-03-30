from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to path if running from within app folder
# (assuming streamlit is run from project root)
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import ButterflyClassifier

# Page configuration
st.set_page_config(
    page_title="Butterfly Classifier",
    page_icon="🦋",
    layout="centered",
)

# Title and description
st.title("🦋 Butterfly Classifier")
st.markdown(
    "Upload a photo of a butterfly, and the model will identify the species. "
    "The model is trained on 8 common butterfly species using iNaturalist images."
)

# Cache the model loader to avoid reloading on every interaction
@st.cache_resource
def load_model() -> ButterflyClassifier:
    return ButterflyClassifier()

# Load model (once)
try:
    classifier = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a butterfly image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Classify", type="primary"):
        with st.spinner("Analyzing..."):
            predictions = classifier.predict(image, top_k=3)

        st.subheader("Predictions")
        for pred in predictions:
            st.write(f"**{pred['label'].replace('_', ' ').title()}** – {pred['confidence']:.2%}")
            st.progress(pred['confidence'])

        # Optional: show top prediction
        top = predictions[0]
        st.success(f"Most likely: **{top['label'].replace('_', ' ').title()}** with {top['confidence']:.2%} confidence")

# Footer
st.markdown("---")
st.caption("Model trained using TensorFlow on iNaturalist data.")
