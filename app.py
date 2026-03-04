import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
from src.classifier import load_model, load_labels, classify

#Page Config
st.set_page_config(
    page_title="Image Classifier",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Image Classifier")
st.markdown("Upload any image and a pretrained ResNet50 model will identify what's in it")

# Load model and labels once
@st.cache_resource
def get_model():
    with st.spinner("Loading model ..."):
        return load_model()
    
@st.cache_resource
def get_labels():
    return load_labels()

model = get_model()
labels = get_labels()

#Upload
uploaded_file = st.file_uploader("Upload an image", type = ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Top 5 Predictions")
        with st.spinner("Classifying..."):
            results = classify(image, model, labels, top_k=5)

        for i, result in enumerate(results):
            st.metric(
                label=f"#{i+1} {result['label']}",
                value=f"{result['confidence']}%"
            )
        
    # Bar chart
    st.subheader("Confidence Scores")
    df = pd.DataFrame(results)
    fig = px.bar(
        df, x="confidence", y="label",
        orientation="h",
        color="confidence",
        color_continuous_scale="blues",
        template="plotly_dark",
        labels={"confidence": "Confidence (%)", "label": "Class"}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload a JPG or PNG image to get started.")