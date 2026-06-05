#!/usr/bin/env python3
"""
capture_auto_40.py

Automatically capture N face images from a USB webcam and save cropped face images
to a folder for a given person. Designed for automatic collection (no keypresses).

Usage example:
  python capture_auto_40.py --name Milan --out ./dataset/raw --count 40 --interval-frames 8

Requirements:
  pip install opencv-python numpy
"""
import cv2
import os
import argparse
import numpy as np
from datetime import datetime

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='Person name / label (folder will be created)')
    p.add_argument('--out', default='./dataset/raw', help='Output base directory')
    p.add_argument('--cam', type=int, default=0, help='Camera index (default 0)')
    p.add_argument('--width', type=int, default=1920, help='Capture width (default 1920)')
    p.add_argument('--height', type=int, default=1080, help='Capture height (default 1080)')
    p.add_argument('--count', type=int, default=40, help='Number of face images to capture (default 40)')
    p.add_argument('--interval-frames', type=int, default=8, help='Check/save every N frames (default 8)')
    p.add_argument('--min-diff', type=float, default=8.0, help='Min average pixel diff to accept new image (default 8.0)')
    p.add_argument('--align', action='store_true', help='Try simple eye-based alignment before saving (may fail sometimes)')
    return p.parse_args()

def detect_faces(gray, face_cascade):
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

def detect_eyes(gray, eye_cascade):
    return eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) if eye_cascade is not None else []

def align_face(color_img, face_rect, eye_cascade):
    x,y,w,h = face_rect
    face_gray = cv2.cvtColor(color_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    eyes = detect_eyes(face_gray, eye_cascade)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        eye_centers = [(ex + ew//2, ey + eh//2) for (ex,ey,ew,eh) in eyes]
        eye_centers = [(x+cx, y+cy) for (cx,cy) in eye_centers]
        (x1,y1), (x2,y2) = eye_centers[:2]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((x1+x2)//2, (y1+y2)//2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        h_img, w_img = color_img.shape[:2]
        rotated = cv2.warpAffine(color_img, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
        rx, ry, rw, rh = x, y, w, h
        return rotated[ry:ry+rh, rx:rx+rw]
    else:
        return color_img[y:y+h, x:x+w]

def avg_pixel_diff(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return float(np.mean(diff))

def main():
    args = parse_args()
    out_dir = os.path.join(args.out, args.name)
    ensure_dir(out_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') if args.align else None

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera index", args.cam)
        return

    saved = 0
    frame_idx = 0
    last_saved_face = None
    print(f"Starting automatic capture for '{args.name}': target {args.count} images.")
    print("Press Ctrl+C to abort or focus the preview window and press 'q' to quit.")

    try:
        while saved < args.count:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: failed to read frame")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, face_cascade)

            if len(faces) > 0:
                faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x,y,w,h = faces_sorted[0]
                cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)

                if frame_idx % args.interval_frames == 0:
                    if args.align and eye_cascade is not None:
                        crop = align_face(frame, (x,y,w,h), eye_cascade)
                        if crop is None or crop.size == 0:
                            crop = frame[y:y+h, x:x+w]
                    else:
                        crop = frame[y:y+h, x:x+w]

                    if crop.size == 0:
                        frame_idx += 1
                        continue

                    try:
                        cmp = cv2.resize(crop, (128,128))
                    except Exception:
                        frame_idx += 1
                        continue

                    if last_saved_face is None:
                        accept = True
                        diff_val = None
                    else:
                        diff_val = avg_pixel_diff(cmp, last_saved_face)
                        accept = diff_val >= args.min_diff

                    if accept:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = os.path.join(out_dir, f"{args.name}_{saved:03d}_{ts}.jpg")
                        cv2.imwrite(fname, crop)
                        saved += 1
                        last_saved_face = cmp.copy()
                        print(f"[{saved}/{args.count}] Saved: {fname}" + (f"  (diff={diff_val:.2f})" if diff_val is not None else ""))

            cv2.putText(display, f"Saved: {saved}/{args.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow("Auto Capture", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual quit.")
                break

            frame_idx += 1

        print("Capture complete. Total saved:", saved)

    except KeyboardInterrupt:
        print("\nAborted by user. Total saved:", saved)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
