from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import time
import json
import os
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inference")

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)

# ---- Model ----
model_path = os.path.join(BASE_DIR, "Finetuned_Model.keras")
model = tf.keras.models.load_model(model_path)

# ---- Supabase ----
TABLE_NAME = "disease_pesticide_recommendations"
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

CLASS_NAMES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Background_without_leaves", "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grapes___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Paddy___Bacterialblight", "Paddy___Blast", "Paddy___Brownspot", "Paddy___Tungro", "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Bacteria", "Potato___Fungi", "Potato___Late_blight", "Potato___Nematode", "Potato___Pest", "Potato___Virus", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Sugercane___Mosaic", "Sugercane___RedRot", "Sugercane___Rust", "Sugercane___Yellow", "Sugercane___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy", "Watermelon___anthracnose", "Watermelon___downy_mildew", "Watermelon___healthy", "Watermelon___mosaic_virus"]

# Fail loudly at startup if the model and class list ever drift apart again
assert model.output_shape[-1] == len(CLASS_NAMES), \
    f"Model has {model.output_shape[-1]} outputs but CLASS_NAMES has {len(CLASS_NAMES)}"

CONFIDENCE_THRESHOLD = 60.0
CACHE_TTL = 600
_cache = {"data": {}, "loaded_at": 0.0}


def get_knowledge():
    if supabase and time.time() - _cache["loaded_at"] > CACHE_TTL:
        try:
            rows = supabase.table(TABLE_NAME).select("*").execute().data
            _cache["data"] = {r["class_name"]: r for r in rows}
            missing = [c for c in CLASS_NAMES if c not in _cache["data"]]
            logger.info(json.dumps({"event": "knowledge_loaded",
                                    "rows": len(rows),
                                    "classes_without_row": missing}))
        except Exception:
            logger.exception("Failed to load knowledge base")
        _cache["loaded_at"] = time.time()
    return _cache["data"]


def pretty_names(class_name):
    # Naming is now consistent ("___" everywhere), so the single-underscore
    # special-casing from before is no longer needed.
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop, disease = "", class_name
    return crop.replace("_", " "), disease.replace("_", " ")


@app.get("/", response_class=HTMLResponse)
def homepage():
    with open(os.path.join(BASE_DIR, "test_predict.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image = image.resize((380, 380))
    img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    probs = model.predict(img_array)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"class_name": CLASS_NAMES[i], "confidence_percentage": round(float(probs[i]) * 100, 2)}
            for i in top3_idx]

    predicted_idx = int(top3_idx[0])
    class_name = CLASS_NAMES[predicted_idx]
    confidence = top3[0]["confidence_percentage"]
    crop, disease_label = pretty_names(class_name)

    # Decide status — restored the no_leaf branch since
    # "Background_without_leaves" is back in the class list
    if class_name == "Background_without_leaves":
        status = "no_leaf"
    elif confidence < CONFIDENCE_THRESHOLD:
        status = "uncertain"
    elif class_name.lower().endswith("healthy"):
        status = "healthy"
    else:
        status = "diseased"

    response = {
        "status": status,
        "class_index": predicted_idx,
        "class_name": class_name,
        "crop": crop,
        "disease": disease_label,
        "confidence_percentage": confidence,
        "top3": top3,
        "symptoms": None,
        "management": None,
        "source": None,
    }

    if status == "diseased":
        row = get_knowledge().get(class_name) or {}
        response["disease"] = row.get("disease") or disease_label
        response["symptoms"] = row.get("symptoms") or "Symptom information is not available yet."
        response["management"] = row.get("management") or "Management information is not available yet."
        response["source"] = row.get("source")

    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(json.dumps({
        "event": "prediction",
        "filename": file.filename,
        "predicted_class": class_name,
        "status": status,
        "confidence_percentage": confidence,
        "has_management_info": bool(response["management"]) and status == "diseased",
        "duration_ms": duration_ms,
    }))

    return response
