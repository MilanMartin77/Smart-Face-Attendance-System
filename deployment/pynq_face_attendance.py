#!/usr/bin/env python3
"""
pynq_face_attendance.py — FaceNet TFLite inference on PYNQ-Z2

Real-time face recognition with attendance logging.
Works in both display (HDMI) and headless (SSH/Jupyter) modes.

Usage:
    python pynq_face_attendance.py \
        --model  model_dynamic.tflite \
        --gallery gallery.npy \
        --names   names.txt \
        --save-log attendance_log.csv

    # Headless (no display, print-only):
    python pynq_face_attendance.py --model ... --headless
"""

import argparse
import os
import time
import datetime
import traceback
import cv2
import numpy as np
import signal

# ── Graceful Ctrl+C ──────────────────────────────────────────────────────────
STOP_REQUESTED = False

def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n🛑 Ctrl+C detected — stopping...")

signal.signal(signal.SIGINT, handle_sigint)


# ── TFLite runtime (prefer lightweight tflite_runtime on PYNQ) ───────────────
def get_interpreter_class():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter, "tflite_runtime"
    except Exception:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow.lite"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_gallery_and_names(gpath, names_path):
    gallery = np.load(gpath).astype(np.float32)
    gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)
    with open(names_path, 'r') as f:
        names = [l.strip() for l in f if l.strip()]
    return gallery, names


def prepare_input(img_rgb, size, input_details):
    """
    Resize, apply FaceNet standardisation, and quantise if needed.
    img_rgb : HxWx3 uint8 RGB image
    """
    x = cv2.resize(img_rgb, (size, size)).astype(np.float32)
    x = (x - 127.5) / 128.0            # FaceNet: scale to [-1, 1]
    x = np.expand_dims(x, 0)           # → (1, H, W, 3)

    dtype = input_details['dtype']
    if dtype in [np.uint8, np.int8]:
        scale, zero = input_details.get('quantization', (1.0, 0))
        scale = scale if scale != 0 else 1.0
        q = np.round(x / scale + zero).astype(dtype)
        q = np.clip(q, -128, 127) if dtype == np.int8 else np.clip(q, 0, 255)
        return q
    return x.astype(dtype)


def dequantize_output(out_tensor, output_details):
    dtype = output_details['dtype']
    if dtype in [np.uint8, np.int8]:
        scale, zero = output_details.get('quantization', (1.0, 0))
        scale = scale if scale != 0 else 1.0
        return (out_tensor.astype(np.float32) - zero) * scale
    return out_tensor.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="PYNQ-Z2 Face Attendance (FaceNet TFLite)")
    p.add_argument('--model',    required=True,        help='Path to .tflite model')
    p.add_argument('--gallery',  required=True,        help='Path to gallery.npy')
    p.add_argument('--names',    required=True,        help='Path to names.txt')
    p.add_argument('--cam',      default=0,            help='Camera index or RTSP URL')
    p.add_argument('--size',     type=int, default=160, help='FaceNet input size (default 160)')
    p.add_argument('--th',       type=float, default=0.60, help='Cosine similarity threshold')
    p.add_argument('--cascade',  default=None,         help='Path to Haar cascade XML')
    p.add_argument('--save-log', default=None,         help='CSV path for attendance log')
    p.add_argument('--headless', action='store_true',  help='No display — terminal output only')
    args = p.parse_args()

    # Validate paths
    for label, path in [("Model", args.model), ("Gallery", args.gallery), ("Names", args.names)]:
        if not os.path.exists(path):
            raise SystemExit(f"{label} not found: {path}")

    # Find Haar cascade (board-specific paths + fallback)
    if args.cascade is None:
        for cp in [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
        ]:
            if os.path.exists(cp):
                args.cascade = cp
                break
        if args.cascade is None:
            raise SystemExit("Haar cascade XML not found. Use --cascade <path>")

    print(f"Cascade: {args.cascade}")

    # Load everything
    InterpreterClass, runtime_name = get_interpreter_class()
    print(f"Runtime: {runtime_name}")

    gallery, names = load_gallery_and_names(args.gallery, args.names)
    print(f"✅ Gallery: {gallery.shape}  |  Names: {names}")

    interp = InterpreterClass(model_path=args.model)
    interp.allocate_tensors()
    input_details  = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]

    print(f"✅ Model: {args.model}")
    print(f"   Input : {input_details['shape']}  dtype={input_details['dtype']}")
    print(f"   Output: {output_details['shape']} dtype={output_details['dtype']}")

    face_cascade = cv2.CascadeClassifier(args.cascade)

    # Camera
    try:
        cam_src = int(args.cam)
        cap = cv2.VideoCapture(cam_src, cv2.CAP_V4L2)
    except ValueError:
        cap = cv2.VideoCapture(args.cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.cam}")

    print("✅ Camera opened")
    print("\n🎥 Recognition started (Ctrl+C to quit)\n")

    last_seen   = {}
    frame_count = 0

    try:
        while not STOP_REQUESTED:
            ret, frame = cap.read()
            if STOP_REQUESTED:
                break
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                crop = frame[y:y+h, x:x+w]
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                try:
                    inp = prepare_input(rgb, args.size, input_details)
                except Exception as ex:
                    print(f"⚠️  Prepare input error: {ex}")
                    continue

                interp.set_tensor(input_details['index'], inp)
                interp.invoke()
                out = interp.get_tensor(output_details['index']).squeeze()

                emb = dequantize_output(out, output_details)
                emb = emb / (np.linalg.norm(emb) + 1e-10)

                sims       = gallery.dot(emb)
                best       = int(np.argmax(sims))
                best_sim   = float(sims[best])

                if best_sim >= args.th:
                    recognized = names[best]
                    label      = f"{recognized} ({best_sim:.2f})"
                    color      = (0, 255, 0)
                else:
                    recognized = None
                    label      = f"Unknown ({best_sim:.2f})"
                    color      = (0, 0, 255)

                if not args.headless:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {ts}  {label}")

                # Attendance log (deduplicated: 5-second cooldown per person)
                if args.save_log and recognized:
                    now  = time.time()
                    last = last_seen.get(recognized, 0)
                    if now - last > 5.0:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(args.save_log, 'a') as logf:
                            logf.write(f'{ts},{recognized},{best_sim:.3f}\n')
                        last_seen[recognized] = now
                        print(f"  ✅  Logged: {recognized}  sim={best_sim:.3f}  @ {ts}")

            if not args.headless:
                cv2.imshow("PYNQ Face Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit.")
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"  [{frame_count} frames processed]")
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception:
        print("\n❌ Exception in main loop:")
        traceback.print_exc()
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\n✅ Cleanup complete")


if __name__ == '__main__':
    main()
