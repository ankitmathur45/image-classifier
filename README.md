markdown# 👁️ Image Classifier

An interactive image classification app built with Streamlit and PyTorch.
Upload any image and a pretrained ResNet50 model identifies what's in it — with confidence scores and visual results.

## Features

- 🖼️ Upload any JPG or PNG image
- 🤖 ResNet50 pretrained on ImageNet (1000 classes)
- 📊 Top 5 predictions with confidence scores
- 📈 Interactive confidence bar chart

## Setup

```bash
git clone git@github.com:ankitmathur45/image-classifier.git
cd image-classifier
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Python 3.12
- PyTorch + TorchVision
- Streamlit
- Plotly
- Pillow
