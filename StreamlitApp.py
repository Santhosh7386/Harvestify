# StreamlitApp.py
# Plant Disease Classifier (Keras) — deploy-safe (no st.stop() inside caches)

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st

# ---- Import TF/Keras early to surface import issues clearly ----
import tensorflow as tf
import keras

# =================== PATHS (edit if needed) ===================
MODEL_PATH = Path("trained_plant_disease_model.keras")   # or Path("models")/"trained_plant_disease_model.keras"
CLASS_MAP_PATH = Path("class_indices.json")              # or Path("models")/"class_indices.json"
# ===============================================================

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier (Keras)")

# ----------------- Helpers -----------------

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
    Returns (x, preview_img)
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

# ----------------- Cached loaders (no st.stop inside) -----------------

@st.cache_resource
def load_class_names_cached(path: Path):
    result = {"ok": False, "names": None, "error": "", "path": str(path)}
    if not path.exists():
        result["error"] = f"class_indices.json not found at: {path}"
        return result
    try:
        with path.open("r") as f:
            mapping = json.load(f)  # {"0":"...", "1":"...", ...}
        mapping = {int(k): v for k, v in mapping.items()}
        names = [mapping[i] for i in sorted(mapping.keys())]
        result.update(ok=True, names=names)
    except Exception as e:
        result["error"] = f"Failed to read/parse class_indices.json: {e}"
    return result

@st.cache_resource
def load_model_cached(path: Path):
    """
    Try load as file or directory SavedModel. Do not call st.stop() here.
    Return dict with ok, model or error, plus some diagnostics.
    """
    here = Path.cwd()
    diag_files = []
    try:
        diag_files = sorted(os.listdir(here))[:50]
    except Exception:
        diag_files = ["<could not list cwd>"]

    result = {
        "ok": False,
        "model": None,
        "error": "",
        "path": str(path),
        "cwd": str(here),
        "files": diag_files,
        "tf": tf.__version__,
        "keras": keras.__version__,
    }

    if not path.exists():
        result["error"] = f"Model path not found: {path}"
        return result

    load_target = str(path)
    try:
        model = tf.keras.models.load_model(load_target, compile=False)
        result.update(ok=True, model=model)
        return result
    except Exception as e1:
        # try compile=True as fallback
        try:
            model = tf.keras.models.load_model(load_target, compile=True)
            result.update(ok=True, model=model)
            return result
        except Exception as e2:
            result["error"] = (
                "Keras failed to load the model.\n"
                f"Attempt 1 (compile=False): {e1}\n\n"
                f"Attempt 2 (compile=True): {e2}"
            )
            return result

# ----------------- Main flow -----------------

# Show runtime info (helpful during deploy)
st.caption(
    f"Python: {os.sys.version.split()[0]} | TF: {tf.__version__} | Keras: {keras.__version__}"
)

# Load labels
labels_res = load_class_names_cached(CLASS_MAP_PATH)
if not labels_res["ok"]:
    st.error(labels_res["error"])
    st.write(f"Checked path: {labels_res['path']}")
    st.stop()
class_names = labels_res["names"]

# Load model
model_res = load_model_cached(MODEL_PATH)
if not model_res["ok"]:
    st.error("❌ Could not load model.")
    st.write(f"Path: {model_res['path']}")
    st.write(f"CWD: {model_res['cwd']}")
    st.write(f"Files here: {model_res['files']}")
    st.write(model_res["error"])
    st.stop()
model = model_res["model"]

# Confirm shapes and alignment
in_shape = model.input_shape      # (None, H, W, 3) typically
out_shape = model.output_shape    # (None, num_classes)

# Some models are dynamic; guard with defaults
try:
    H = int(in_shape[1]) if in_shape[1] is not None else 224
    W = int(in_shape[2]) if in_shape[2] is not None else 224
except Exception:
    H, W = 224, 224

num_model_classes = out_shape[-1] if isinstance(out_shape, tuple) else out_shape[0][-1]
num_label_classes = len(class_names)

st.caption(f"Model expects roughly: {H}×{W}×3 | Outputs: {num_model_classes} classes")

if num_model_classes != num_label_classes:
    st.error(
        f"❌ Label mismatch: model outputs {num_model_classes} classes, "
        f"but JSON has {num_label_classes}. "
        "Update class_indices.json to match the training indices (0..N-1)."
    )
    st.stop()

uploaded = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded)
    x, preview = preprocess(pil, W, H)

    # Predict safely
    try:
        raw = model.predict(x, verbose=0)
    except Exception as e:
        st.error(f"❌ model.predict failed: {e}")
        st.stop()

    probs = to_probs(raw)  # (num_classes,)
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])

    # Show results
    st.image(preview, caption="Uploaded image", use_container_width=True)
    st.subheader("Predicted Disease")
    st.success(f"{top_label} · Confidence: {top_conf:.2%}")

    with st.expander("Top-5 (quick view)"):
        top5 = np.argsort(-probs)[:5]
        for i in top5:
            st.write(f"- {class_names[i]}: {probs[i]*100:.2f}%")

    with st.expander("All classes (ranked)"):
        order = np.argsort(-probs)
        for i in order:
            st.write(f"{class_names[i]} — {probs[i]*100:.2f}%")
