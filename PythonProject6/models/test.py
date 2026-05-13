import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. إعدادات المسارات (Paths Configuration)
# ==========================================
unet_path = r'D:\PythonProject6\PythonProject6\models\Attention_UNet_HCC_with_ID.keras'
feature_extractor_path = r'D:\PythonProject6\PythonProject6\models\Hybrid_FeatureExtractor_V2.h5'
rf_model_path = r'D:\PythonProject6\PythonProject6\models\Hybrid_RandomForest_BestModel_V2.pkl'
data_dir = r'C:\Users\Admin\Downloads\FINAL_DATA_WITH_ID\kaggle\working\FINAL_DATA_WITH_ID'
volumes_dir = os.path.join(data_dir, 'volumes')

# ==========================================
# 2. تحميل منظومة SURGENIUS
# ==========================================
print("⏳ جاري تحميل النماذج...")
try:
    unet_model = load_model(unet_path, compile=False)
    feature_extractor = load_model(feature_extractor_path, compile=False)
    rf_model = joblib.load(rf_model_path)
    print("✅ تم تحميل جميع النماذج بنجاح!\n")
except Exception as e:
    print(f"⚠️ خطأ في تحميل النماذج: {e}")
    exit()

# ==========================================
# 3. رحلة البحث عن ورم (Looping through images)
# ==========================================
image_files = [f for f in os.listdir(volumes_dir) if f.endswith('.png')]

if not image_files:
    print("❌ لم يتم العثور على أي صور في المجلد المحدد.")
    exit()

print(f"🔍 جاري البحث في {len(image_files)} صورة عن أول حالة تحتوي على ورم...\n")

tumor_found = False

for image_name in image_files:
    image_path = os.path.join(volumes_dir, image_name)

    # قراءة الصورة وتجهيزها للـ U-Net
    img_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_input = (img_resized / 255.0).astype(np.float32)

    # التقطيع (Segmentation)
    pred_unet = unet_model.predict(np.expand_dims(img_input, axis=0), verbose=0)[0]
    pred_mask = np.argmax(pred_unet, axis=-1)

    # هل يوجد ورم؟
    pred_has_tumor = 2 in np.unique(pred_mask)

    if pred_has_tumor:
        tumor_found = True
        print("-" * 50)
        print("🚨 تم اكتشاف ورم!")
        print(f"📁 اسم الصورة: {image_name}")
        print("⏳ جاري تحليل الخصائص وتحديد المرحلة...")

        # تلوين الورم وتجهيزه للـ EfficientNet
        highlighted_img = (img_input * 255).astype(np.uint8).copy()
        highlighted_img[pred_mask == 2] = [255, 0, 0]

        img_224 = cv2.resize(highlighted_img, (224, 224))
        img_input_eff = preprocess_input(np.expand_dims(img_224.astype(np.float32), axis=0))

        # استخراج الخصائص
        features = feature_extractor.predict(img_input_eff, verbose=0)

        # التصنيف النهائي
        stage_num = rf_model.predict(features)[0]
        confidence = float(np.max(rf_model.predict_proba(features)[0]) * 100)

        stage_labels = {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"}
        predicted_stage = stage_labels.get(stage_num, f"Stage {stage_num}")

        # التقرير النهائي
        print(f"📊 التشخيص النهائي: {predicted_stage}")
        print(f"🎯 نسبة الثقة: {confidence:.2f}%")
        print("-" * 50)

        # التوقف عن البحث بمجرد إيجاد أول ورم
        break
    else:
        # طباعة بسيطة لمعرفة تقدم الكود (اختياري، يمكنك حذف هذا السطر لو أزعجك)
        print(f"⏩ تخطي الصورة: {image_name} (سليمة)")

if not tumor_found:
    print("✅ تم فحص جميع الصور المتاحة، ولم يتم العثور على أي ورم في المجلد بأكمله!")