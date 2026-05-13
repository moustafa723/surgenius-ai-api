import io
import cv2
import numpy as np
import joblib
import uvicorn  # 🚨 ضفنا دي عشان السيرفر يشتغل من PyCharm مباشرة
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. Initialize FastAPI Application
# ==========================================
app = FastAPI(title="SURGENIUS AI API", version="1.0")

# ==========================================
# 2. Add CORS Middleware
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. Load Models Globally
# ==========================================
print("⏳ Loading SURGENIUS AI Pipeline from Local Paths...")
try:
    unet_path = r'D:\PythonProject6\PythonProject6\models\Attention_UNet_HCC_with_ID.keras'
    feature_extractor_path = r'D:\PythonProject6\PythonProject6\models\Hybrid_FeatureExtractor_V2.h5'
    rf_model_path = r'D:\PythonProject6\PythonProject6\models\Hybrid_RandomForest_BestModel_V2.pkl'

    unet_model = load_model(unet_path, compile=False)
    feature_extractor = load_model(feature_extractor_path, compile=False)
    rf_model = joblib.load(rf_model_path)
    print("✅ All 3 Models Loaded Successfully! Server is Ready. 🚀")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")


# ==========================================
# 4. Define the Prediction Endpoint
# ==========================================
@app.post("/api/v1/analyze-scan")
async def analyze_scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_input = (img_resized / 255.0).astype(np.float32)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    # U-Net Segmentation
    pred_unet = unet_model.predict(np.expand_dims(img_input, axis=0), verbose=0)[0]
    pred_mask = np.argmax(pred_unet, axis=-1)

    pred_has_tumor = 2 in np.unique(pred_mask)

    if not pred_has_tumor:
        return {
            "status": "success",
            "tumor_detected": False,
            "diagnosis": "No Tumor",
            "confidence": 0.0
        }

    # EfficientNet + Random Forest Classification
    highlighted_img = (img_input * 255).astype(np.uint8).copy()
    highlighted_img[pred_mask == 2] = [255, 0, 0]

    img_224 = cv2.resize(highlighted_img, (224, 224))
    img_input_eff = preprocess_input(np.expand_dims(img_224.astype(np.float32), axis=0))

    features = feature_extractor.predict(img_input_eff, verbose=0)

    stage_num = rf_model.predict(features)[0]
    confidence = float(np.max(rf_model.predict_proba(features)[0]) * 100)

    stage_labels = {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"}
    predicted_stage = stage_labels.get(stage_num, f"Stage {stage_num}")

    return {
        "status": "success",
        "tumor_detected": True,
        "diagnosis": predicted_stage,
        "confidence": round(confidence, 2)
    }


# ==========================================
# 5. تشغيل السيرفر تلقائياً
# ==========================================
if __name__ == "__main__":
    print("🌐 جاري تشغيل السيرفر المحلي...")
    uvicorn.run(app, host="0.0.0.0", port=8000)