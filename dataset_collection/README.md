# 📷 Dataset Collection

This folder contains the script used to automatically capture face images for each team member using a USB webcam.

## Script: `capture_auto_40.py`

### How It Works

1. Opens the webcam and detects faces using OpenCV's Haar Cascade classifier
2. Every N frames, checks if the new face is sufficiently different from the last saved image (pixel-diff threshold)
3. Saves the cropped face image as a `.jpg` to `./dataset/raw/<name>/`
4. Stops automatically once the target count is reached

### Requirements

```bash
pip install opencv-python numpy
```

### Usage

```bash
# Basic — capture 40 images for a person named "Alice"
python capture_auto_40.py --name Alice

# Custom count and interval
python capture_auto_40.py --name Alice --count 40 --interval-frames 8

# With eye-based face alignment
python capture_auto_40.py --name Alice --align

# Use a different camera (e.g., camera index 1)
python capture_auto_40.py --name Alice --cam 1

# Full example with all options
python capture_auto_40.py \
  --name Alice \
  --out ./dataset/raw \
  --cam 0 \
  --width 1920 \
  --height 1080 \
  --count 40 \
  --interval-frames 8 \
  --min-diff 8.0 \
  --align
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | _(required)_ | Person label; a subfolder with this name is created |
| `--out` | `./dataset/raw` | Base output directory |
| `--cam` | `0` | Camera index |
| `--width` | `1920` | Capture width in pixels |
| `--height` | `1080` | Capture height in pixels |
| `--count` | `40` | Number of images to capture |
| `--interval-frames` | `8` | Save a candidate every N frames |
| `--min-diff` | `8.0` | Min average pixel difference to accept a new image |
| `--align` | `False` | Enable eye-based face alignment |

### Tips for a Good Dataset

- **Lighting:** Capture in consistent, well-lit conditions — avoid harsh shadows on the face
- **Variation:** Slightly vary head angle, expression, and position between captures
- **Distance:** Stay roughly 0.5–1 m from the camera
- **Background:** Plain or consistent backgrounds reduce noise
- **Glasses:** Capture both with and without glasses if applicable

### Output Structure

After running for all 3 team members:

```
dataset/
└── raw/
    ├── Alice/
    │   ├── Alice_000_20250101_120000.jpg
    │   ├── Alice_001_20250101_120001.jpg
    │   └── ...  (~40 images)
    ├── Bob/
    │   └── ...  (~40 images)
    └── Carol/
        └── ...  (~40 images)
```

### Next Step

Zip the `dataset/raw/` folder and upload to Google Drive for training in Colab:

```bash
zip -r dataset_raw.zip dataset/raw/
```
