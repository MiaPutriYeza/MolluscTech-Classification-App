import os
import base64
import tempfile
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = Flask(__name__)


# --- 1. INISIALISASI MODEL (MOBILENETV2) ---
print("Lagi manggil satpam MobileNetV2...")
satpam_model = MobileNetV2(weights='imagenet')
print("Model MobileNetV2")

# --- 2. INISIALISASI MODEL KERANG (TFLITE) ---
interpreter = tf.lite.Interpreter(model_path="model_kerang.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ['batik', 'bulu', 'dara', 'hijau', 'kepah']
THRESHOLD = 50.0 

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    
    # TAHAP 1: CEK \MOBILENETV2
    img_satpam = np.array(img_resized, dtype=np.float32)
    img_satpam = np.expand_dims(img_satpam, axis=0)
    img_satpam = preprocess_input(img_satpam)
    
    # Prediksi pake MobileNet
    preds_satpam = satpam_model.predict(img_satpam)
    top_pred = decode_predictions(preds_satpam, top=1)[0][0]
    satpam_label = top_pred[1]
    satpam_conf = top_pred[2] * 100
    
    # Print 
    print(f"[LOG SATPAM] Gambar dideteksi sebagai: {satpam_label} ({satpam_conf:.2f}%)")
    
    # ImageNet punya beberapa class yg mirip/berkaitan sama kerang. 
    # Kalo tebakan satpam NGGA ADA di list ini, langsung kita tolak!
    kerang_keywords = ['conch', 'chambered_nautilus', 'snail', 'slug', 'chiton', 'coral', 'rock', 'stone', 'crab', 'isopod']
    
    is_kerang_or_sea_stuff = any(keyword in satpam_label.lower() for keyword in kerang_keywords)
    
    # Kalo bukan kerang/batu dan model yakin banget (>30%),
    if not is_kerang_or_sea_stuff and satpam_conf > 30.0:
        return "Ini bukan kerang!", 0.0, None

    # TAHAP 2: MASUK KE MODEL KERANG
    # Preprocessing TFLite (dibagi 255.0)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0].copy()
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    class_probs = [
        {"label": LABELS[i], "prob": float(prediction[i]) * 100}
        for i in range(len(LABELS))
    ]
    
    if confidence < THRESHOLD:
        return "Ini bukan kerang!", confidence, class_probs
        
    return LABELS[class_idx], confidence, class_probs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Pilih file terlebih dahulu'})
        
        if file:
            filename = secure_filename(file.filename)
            suffix = os.path.splitext(filename)[1] or ".jpg"
            image_bytes = file.read()

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(image_bytes)
                    temp_path = temp_file.name

                label, confidence, class_probs = predict_image(temp_path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            mime_type = file.mimetype or "image/jpeg"
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            image_url = f"data:{mime_type};base64,{image_b64}"

            return render_template(
                'index.html',
                label=label,
                confidence=confidence,
                image_url=image_url,
                class_probs=class_probs,
            )
    
    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)