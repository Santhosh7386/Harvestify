# StreamlitApp.py
# Plant Disease Classifier (Keras) — robust loader + full ranked output

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st

# ---- TF/Keras imports (do early to surface version issues) ----
try:
    import tensorflow as tf
    import keras
    from tensorflow.keras.models import load_model as tf_load_model
except Exception as e:
    st.error(f"❌ TensorFlow/Keras import failed: {e}")
    raise

# =================== CONFIG ===================
# If your files are in the repo root:
MODEL_PATH = Path("trained_plant_disease_model.keras")   # or Path("models")/"trained_plant_disease_model.keras"
CLASS_MAP_PATH = Path("class_indices.json")              # or Path("models")/"class_indices.json"
# ==============================================

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier (Keras • robust)")

# ----------------- Utilities ------------------

def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def preprocess(pil_img: Image.Image, target_w: int, target_h: int):
    """
    Center-crop -> resize -> scale [0,1] -> add batch dim.
    Returns: (np.array (1,H,W,3), preview PIL)
    """
    img = pil_img.convert("RGB")
    img = center_crop_to_square(img)
    img = img.resize((target_w, target_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr, img

def to_probs(pred):
    """Ensure a 1D probability vector (apply softmax if needed)."""
    pred = np.asarray(pred)
    vec = pred[0] if pred.ndim == 2 and pred.shape[0] == 1 else pred
    s = np.sum(vec)
    if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
        e = np.exp(vec - np.max(vec))
        vec = e / np.sum(e)
    return vec

# ----------------- Loaders --------------------

@st.cache_resource
def load_class_names():
    if not CLASS_MAP_PATH.exists():
        st.error(f"❌ class_indices.json not found at: {CLASS_MAP_PATH}")
        st.stop()
    with CLASS_MAP_PATH.open("r") as f:
        mapping = json.load(f)  # {"0":"...", "1":"...", ...}
    try:
        mapping = {int(k): v for k, v in mapping.items()}
    except Exception as e:
        st.error(f"❌ class_indices.json keys must be numeric strings ('0','1',...). Error: {e}")
        st.stop()
    names = [mapping[i] for i in sorted(mapping.keys())]
    return names

@st.cache_resource
def load_cls_model():
    # Helpful diagnostics in UI
    st.caption(
        f"Runtime → Python: {os.sys.version.split()[0]} | "
        f"TF: {tf.__version__} | Keras: {keras.__version__}"
    )
    here = Path.cwd()
    st.caption(f"CWD: {here} | Looking for model at: {MODEL_PATH}")

    # 1) Existence check
    if not MODEL_PATH.exists():
        try:
            files_here = ", ".join(sorted(os.listdir(here))[:50])
        except Exception:
            files_here = "(could not list)"
        st.error(
            "❌ Model file/folder not found.\n\n"
            f"Expected at: {MODEL_PATH}\n\n"
            f"Files in CWD: {files_here}\n\n"
            "➡️ Fix path or place the model at that location."
        )
        st.stop()

    # 2) File vs directory
    load_target = str(MODEL_PATH)
    fmt = "SavedModel directory" if MODEL_PATH.is_dir() else "single file"

    # 3) Try compile=False first (avoids training config issues)
    try:
        model = tf.keras.models.load_model(load_target, compile=False)
        return model
    except Exception as e1:
        # 4) Fallback to compile=True to handle some saved configs
        try:
            model = tf.keras.models.load_model(load_target, compile=True)
            return model
        except Exception as e2:
            st.error(
                "❌ Keras failed to load the model.\n\n"
                f"Path: {MODEL_PATH} ({fmt})\n\n"
                f"Attempt 1 (compile=False) error:\n{e1}\n\n"
                f"Attempt 2 (compile=True) error:\n{e2}\n\n"
                "Common fixes:\n"
                "• Confirm path (file vs folder)\n"
                "• Use a compatible TF/Keras version (match how the model was saved)\n"
                "• If it’s a directory model, point MODEL_PATH to the folder itself"
            )
            st.stop()

# ----------------- Main App -------------------

class_names = load_class_names()
model = load_cls_model()

# Confirm shapes and alignment
in_shape = model.input_shape      # e.g., (None, H, W, 3)
out_shape = model.output_shape    # e.g., (None, num_classes)

try:
    H = int(in_shape[1]) if in_shape[1] is not None else 224
    W = int(in_shape[2]) if in_shape[2] is not None else 224
except Exception:
    H, W = 224, 224  # fallback if model is flexible/dynamic
num_model_classes = out_shape[-1] if isinstance(out_shape, tuple) else out_shape[0][-1]
num_label_classes = len(class_names)

st.caption(f"Model expects roughly: {H}×{W}×3 | Outputs: {num_model_classes} classes")
if num_model_classes != num_label_classes:
    st.error(
        f"❌ Label mismatch:\n"
        f"Model outputs {num_model_classes} classes, but JSON has {num_label_classes}.\n"
        f"➡️ Update class_indices.json to match the model's training indices."
    )
    st.stop()

uploaded = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded)
    x, preview = preprocess(pil, W, H)

    # Predict
    try:
        raw = model.predict(x, verbose=0)
    except Exception as e:
        st.error(f"❌ model.predict failed: {e}")
        st.stop()

    probs = to_probs(raw)   # (num_classes,)
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])

    # Show image + top-1
    st.image(preview, caption="Uploaded image", use_container_width=True)
    st.subheader("Predicted Disease")
    st.success(f"{top_label} · Confidence: {top_conf:.2%}")

    # Top-5 quick view
    with st.expander("Top-5 (quick view)"):
        top5 = np.argsort(-probs)[:5]
        for i in top5:
            st.write(f"- {class_names[i]}: {probs[i]*100:.2f}%")

    # Full ranked list (ALL classes)
    with st.expander("All classes (ranked)"):
        order = np.argsort(-probs)
        for i in order:
            st.write(f"{class_names[i]} — {probs[i]*100:.2f}%")
