"""
pc_test_attendance.py

Test the trained FaceNet TFLite model + gallery on a PC webcam
before deploying to the PYNQ-Z2 board.

Requirements:
    pip install opencv-python numpy tensorflow

Usage:
    python pc_test_attendance.py

Files needed in the same directory:
    model_dynamic.tflite   (or model_int8.tflite)
    gallery.npy
    names.txt
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict

# ================= CONFIG =================
IMG_SIZE        = (160, 160)
MODEL_PATH      = "model_dynamic.tflite"
SIM_THRESHOLD   = 0.60        # cosine similarity threshold for recognition
MARGIN          = 0.08        # expand detected face bbox by this fraction
REQUIRED_FRAMES = 15          # consecutive frames before marking PRESENT
GALLERY_PATH    = "gallery.npy"
NAMES_PATH      = "names.txt"

# ================= LOAD GALLERY =================
gallery = np.load(GALLERY_PATH).astype(np.float32)
gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)

with open(NAMES_PATH, "r") as f:
    names = [line.strip() for line in f.readlines()]

attendance  = {name: "ABSENT"  for name in names}
frame_count = defaultdict(int)

print(f"✅ Loaded gallery: {gallery.shape}")
print(f"   People: {names}")

# ================= LOAD TFLITE MODEL =================
interpreter    = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"   Input shape:  {input_details['shape']}")
print(f"   Output shape: {output_details['shape']}")

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("\n🎥 Camera started. Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.imshow("Face Attendance — PC Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        # Expand bbox with margin
        mh = int(h * MARGIN)
        mw = int(w * MARGIN)
        y1 = max(0, y - mh);  y2 = min(frame.shape[0], y + h + mh)
        x1 = max(0, x - mw);  x2 = min(frame.shape[1], x + w + mw)

        face = frame[y1:y2, x1:x2]

        # Preprocess
        img  = cv2.resize(face, IMG_SIZE)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype(np.float32)
        img  = (img - 127.5) / 128.0           # FaceNet standardisation
        inp  = np.expand_dims(img, axis=0)

        # Handle int8 quantisation (if using model_int8.tflite)
        if input_details["dtype"] == np.int8:
            scale, zero = input_details["quantization"]
            inp = np.round(inp / scale + zero).astype(np.int8)
            inp = np.clip(inp, -128, 127)

        # Inference
        interpreter.set_tensor(input_details["index"], inp)
        interpreter.invoke()
        emb = interpreter.get_tensor(output_details["index"]).squeeze()

        # Dequantise int8 output
        if output_details["dtype"] == np.int8:
            scale, zero = output_details["quantization"]
            emb = (emb.astype(np.float32) - zero) * scale

        emb  = emb / (np.linalg.norm(emb) + 1e-10)

        # Cosine similarity against gallery
        sims      = gallery @ emb
        best_idx  = np.argmax(sims)
        best_score = float(sims[best_idx])

        if best_score > SIM_THRESHOLD:
            name = names[best_idx]
            frame_count[name] += 1
            if frame_count[name] >= REQUIRED_FRAMES:
                attendance[name] = "PRESENT"
            label = f"{name} ({best_score:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({best_score:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Attendance overlay (bottom-left)
    y0 = frame.shape[0] - (len(attendance) * 25) - 10
    for person, status in attendance.items():
        status_color = (0, 255, 0) if status == "PRESENT" else (255, 255, 0)
        cv2.putText(frame, f"{person}: {status}",
                    (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y0 += 25

    cv2.imshow("Face Attendance — PC Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("FINAL ATTENDANCE REPORT")
print("="*50)
for person, status in attendance.items():
    icon = "✅" if status == "PRESENT" else "❌"
    print(f"  {icon}  {person}: {status}")
print("="*50)
