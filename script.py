from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from model import CNN  # Reuse same model structure

app = Flask(__name__)

# Load model
model = CNN()
model.load_state_dict(torch.load("model/fashion_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Class names
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Image transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).convert("L")
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, 1).item()
                prediction = classes[pred]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
