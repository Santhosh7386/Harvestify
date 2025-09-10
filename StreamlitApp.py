import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

# --------------------
# Model definition
# --------------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# --------------------
# Paths
# --------------------
MODEL_PATH = "plant_disease_model.pth"
CLASS_MAP_PATH = "class_indices.json"

# --------------------
# Load class names
# --------------------
with open(CLASS_MAP_PATH, "r") as f:
    mapping = json.load(f)
mapping = {int(k): v for k, v in mapping.items()}
class_names = [mapping[i] for i in sorted(mapping.keys())]

# --------------------
# Load model
# --------------------
num_classes = len(class_names)
model = ResNet9(3, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --------------------
# Preprocessing (try B first)
# --------------------
# Option B: simple scaling (most likely for your ResNet9)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # match training input size
    transforms.ToTensor(),          # converts to [0,1]
])

# If your training used ImageNet normalization, switch to:
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# --------------------
# Streamlit UI
# --------------------
st.title("🌿 Plant Disease Classifier")

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs[0], dim=0)

    # Top-1
    top1 = torch.argmax(probs).item()
    st.subheader("Most Likely Disease")
    st.write(f"🌱 **{class_names[top1]}** ({probs[top1]*100:.2f}%)")

    # Top-3
    st.subheader("Other Possible Diseases")
    top3 = torch.topk(probs, 3)
    for i, idx in enumerate(top3.indices):
        disease = class_names[idx.item()]
        confidence = top3.values[i].item() * 100
        st.write(f"{i+1}. {disease} – {confidence:.2f}%")

