import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

def load_model():
    """Load pretrained ResNet50 model"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def load_labels():
    """Load ImageNet class labels"""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as response:
        labels = json.loads(response.read().decode())
    return labels

def preprocess_image(image: Image.Image):
    """Preprocess image for ResNet50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def classify(image: Image.Image, model, labels, top_k=5):
    """Run inference and return top k predictions"""
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []

    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "label": labels[idx.item()].replace("_", " ").title(),
            "confidence": round(prob.item()*100, 2)
        })
    return results

