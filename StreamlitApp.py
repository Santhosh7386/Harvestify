# app.py
import json
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = "trained_plant_disease_model.keras"
CLASS_MAP_PATH = "class_indices.json"

st.title("🌿 Plant Disease Classifier (Stabilized)")
st.caption("Center-crop + resize; averaged over 3 common preprocessing modes to avoid skew.")

@st.cache_resource
def load_cls_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_MAP_PATH, "r") as f:
        mapping = json.load(f)  # {"0":"...", "1":"...", ...}
    mapping = {int(k): v for k, v in mapping.items()}
    # IMPORTANT: order by numeric index so indices match training outputs
    names = [mapping[i] for i in sorted(mapping.keys())]
    return names

def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def preprocess_variants(pil_img, W, H):
    """Return list of (name, array) for the 3 common preprocessing modes."""
    img = pil_img.convert("RGB")
    img = center_crop_to_square(img)
    img = img.resize((W, H))
    rgb = np.array(img, dtype=np.float32)

    # A) RGB scaled to [0,1]
    a = (rgb / 255.0)[None, ...]

    # B) TF mode [-1, 1]
    b = ((rgb / 127.5) - 1.0)[None, ...]

    # C) Caffe BGR mean subtraction
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    bgr = rgb[:, :, ::-1]          # RGB -> BGR
    c = (bgr - mean)[None, ...]

    return [("rgb_01", a), ("rgb_tf", b), ("bgr_caffe", c)]

def to_probs(logits_like):
    """Turn logits or already-softmaxed output into a probability vector."""
    vec = logits_like[0] if logits_like.ndim == 2 else logits_like
    vec = np.asarray(vec, dtype=np.float32)
    s = vec.sum()
    # If not already probabilities, softmax
    if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
        e = np.exp(vec - np.max(vec))
        vec = e / e.sum()
    return vec

# Load model + labels
model = load_cls_model()
class_names = load_class_names()

# Confirm input/output dimensions
in_shape = model.input_shape  # (None, H, W, C)
out_shape = model.output_shape  # (None, num_classes)
try:
    H, W, C = in_shape[1], in_shape[2], in_shape[3]
except Exception:
    H, W, C = 224, 224, 3  # fallback if model is dynamic

num_model_classes = out_shape[-1] if isinstance(out_shape, tuple) else out_shape[0][-1]
num_label_classes = len(class_names)

st.write(f"**Model classes:** {num_model_classes} | **Labels in JSON:** {num_label_classes}")
if num_model_classes != num_label_classes:
    st.error("❌ Label mismatch: model output units != number of labels in JSON. "
             "Fix class_indices.json to exactly match training indices.")
    st.stop()

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)

    # Build 3 preprocessed batches and run model
    variants = preprocess_variants(img, W, H)
    probs_list = []
    for vname, arr in variants:
        pred = model.predict(arr, verbose=0)
        probs = to_probs(pred)
        probs_list.append(probs)

    # Average probabilities across preprocessing variants (ensemble for stability)
    probs_avg = np.mean(probs_list, axis=0)

    # Top-1 and top-5
    top1 = int(np.argmax(probs_avg))
    top5 = np.argsort(-probs_avg)[:5]

    st.subheader("Predicted Disease")
    st.write(class_names[top1])

    with st.expander("Top-5 (useful to confirm all classes work)"):
        for i in top5:
            st.write(f"- {class_names[i]}: {probs_avg[i]*100:.2f}%")
