# 🖥️ PC Test — Verify Before Board Deployment

Before deploying to the PYNQ-Z2, this folder lets you verify the full inference
pipeline (TFLite model + gallery) on a regular PC webcam.

## Files Needed

Place these (downloaded from Colab artifacts) in this folder:

```
pc_test/
├── pc_test_attendance.py     ← test script
├── model_dynamic.tflite      ← float32 TFLite model  (from Colab)
├── model_int8.tflite         ← int8 model (optional) (from Colab)
├── gallery.npy               ← face embeddings        (from Colab)
└── names.txt                 ← class names            (from Colab)
```

## Requirements

```bash
pip install opencv-python numpy tensorflow
```

## Run

```bash
cd pc_test
python pc_test_attendance.py
```

## What to Expect

- A webcam window opens with a live feed
- Each detected face gets a bounding box:
  - **Green box** → recognised person + similarity score
  - **Red box** → unknown face
- Bottom-left overlay shows real-time attendance status (ABSENT / PRESENT)
- After `REQUIRED_FRAMES = 15` consistent frames, the person is marked **PRESENT**
- Press `q` to quit; final attendance printed to terminal

## Tuning Tips

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIM_THRESHOLD` | 0.60 | Lower → more lenient; Higher → stricter |
| `REQUIRED_FRAMES` | 15 | Frames before marking PRESENT |
| `MARGIN` | 0.08 | Face crop expansion (helps at angles) |

## Next Step

Once working on PC → deploy to PYNQ-Z2 using `deployment/pynq_face_attendance.py`
