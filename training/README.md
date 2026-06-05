# 🧠 Model Training (Google Colab)

This folder contains the training notebook used to build the face recognition model.

## Files

| File | Description |
|------|-------------|
| `train_colab.ipynb` | Main Colab notebook — dataset loading, preprocessing, training, export |

> **Note:** The notebook will be added after training is complete. See the main README for a summary of the training pipeline.

## Training Pipeline (Summary)

1. **Mount Google Drive** — load the zipped dataset
2. **Preprocess** — resize faces, normalise pixel values, label-encode names
3. **Split** — 80/20 train/validation split
4. **Train** — CNN or transfer learning (e.g., MobileNetV2 fine-tuned)
5. **Evaluate** — accuracy, confusion matrix, per-class report
6. **Export** — save as `.h5` (Keras) or `.tflite` (TFLite for PYNQ)

## How to Run

1. Upload `dataset_raw.zip` to your Google Drive
2. Open `train_colab.ipynb` in [Google Colab](https://colab.research.google.com)
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run all cells
5. Download the exported model file to deploy on PYNQ-Z2

## Dataset Format Expected

```
/content/drive/MyDrive/dataset/raw/
    Alice/   ← JPEG images
    Bob/     ← JPEG images
    Carol/   ← JPEG images
```

## Dependencies (Colab — pre-installed)

```
tensorflow >= 2.x
numpy
matplotlib
scikit-learn
opencv-python-headless
```
