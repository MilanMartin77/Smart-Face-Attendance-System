# 🎓 Smart Face Attendance System
### FaceNet × TFLite × PYNQ-Z2 FPGA

> Automated real-time face recognition attendance system — trained on Google Colab using **FaceNet**, deployed on the **PYNQ-Z2** Zynq SoC, with a custom **convolution layer accelerator** designed in Vitis HLS and implemented in Vivado.

---

## 📌 Overview

| Stage | What we did |
|-------|-------------|
| **Dataset** | Auto-captured ~40 face images per person using a USB webcam |
| **Training** | FaceNet (InceptionResNetV2) embeddings computed on Google Colab GPU; gallery saved as `.npy` |
| **Export** | TFLite (float32 + int8) models exported for edge deployment |
| **PC Test** | Full inference pipeline verified on laptop before board deployment |
| **PYNQ-Z2** | TFLite inference on ARM PS; real-time attendance logged to CSV |
| **FPGA Accel.** | Custom 2D convolution layer IP in Vitis HLS; block design in Vivado; bitstream generated |

---

## 🏗️ System Architecture

```
[ USB Webcam ]
      │
      ▼
[ Haar Cascade Face Detector ]   (OpenCV — ARM PS)
      │  cropped face (160×160 RGB)
      ▼
[ FaceNet TFLite Model ]         (ARM PS — tflite_runtime)
      │  512-d L2-normalised embedding
      ▼
[ Cosine Similarity vs Gallery ] (NumPy dot product)
      │  best match + score
      ▼
[ Attendance Logger ]            (CSV + terminal)
      │
      ▼
[ PYNQ-Z2 Output ]               (HDMI display / SSH headless)

── FPGA Accelerator (parallel track) ──────────────────────
[ conv_layer.cpp → Vitis HLS synthesis ]
      │  exported IP
      ▼
[ Vivado Block Design (PS + conv_layer IP + AXI) ]
      │  bitstream
      ▼
[ PYNQ Overlay (Python) ]        (ARM PS controls PL via AXI-Lite)
```

---


## 📁 Repository Structure

```
smart-face-attendance/
│
├── dataset_collection/
│   ├── capture_auto_40.py         ← Auto-capture ~40 face images per person
│   └── README.md
│
├── training/
│   ├── train_facenet_colab.py     ← Colab pipeline: embed → gallery.npy → TFLite export
│   └── README.md
│
├── pc_test/
│   ├── pc_test_attendance.py      ← Verify TFLite + gallery on laptop webcam
│   └── README.md
│
├── deployment/
│   ├── pynq_face_attendance.py    ← Full inference script for PYNQ-Z2
│   ├── requirements_pynq.txt
│   └── README.md
│
├── fpga_accelerator/
│   ├── hls/
│   │   ├── conv_layer.h           ← Data types, dimensions
│   │   ├── conv_layer.cpp         ← Pipelined 2D conv + ReLU (HLS pragmas)
│   │   └── conv_layer_tb.cpp      ← C testbench
│   └── README.md                  ← HLS → Vivado → PYNQ overlay guide
│
├── docs/
│   └── results.md
│
├── assets/
│   └── screenshots/               ← Vivado block design, HLS report, PYNQ output
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Collect Dataset

```bash
pip install opencv-python numpy
python dataset_collection/capture_auto_40.py --name Alice --count 40
python dataset_collection/capture_auto_40.py --name Bob   --count 40
python dataset_collection/capture_auto_40.py --name Carol --count 40
# Output: ./dataset/raw/<name>/*.jpg
```

### 2. Train on Google Colab

1. Zip `dataset/raw/` → upload to Google Drive as `dataset.zip`
2. Open `training/train_facenet_colab.py` in Colab (GPU runtime)
3. Run all cells → downloads `gallery.npy`, `names.txt`, `model_dynamic.tflite`, `model_int8.tflite`

### 3. Test on PC

```bash
cd pc_test
# place gallery.npy, names.txt, model_dynamic.tflite here
pip install opencv-python numpy tensorflow
python pc_test_attendance.py
```

### 4. Deploy on PYNQ-Z2

```bash
# SCP artifacts to board
scp gallery.npy names.txt model_dynamic.tflite pynq_face_attendance.py xilinx@192.168.2.99:/home/xilinx/

# SSH into board and run
ssh xilinx@192.168.2.99  # password: xilinx
pip3 install tflite-runtime opencv-python-headless
python3 pynq_face_attendance.py \
    --model  model_dynamic.tflite \
    --gallery gallery.npy \
    --names   names.txt \
    --save-log attendance_log.csv \
    --headless
```

### 5. FPGA Accelerator (Vitis HLS + Vivado)

See [`fpga_accelerator/README.md`](fpga_accelerator/README.md) for the full HLS → Vivado → PYNQ overlay workflow.

---

## 🔧 Hardware & Software

| Component | Details |
|-----------|---------|
| **Board** | PYNQ-Z2 (Zynq-7020 SoC — ARM Cortex-A9 + Artix-7 FPGA) |
| **Camera** | USB Webcam (UVC) |
| **Face model** | FaceNet (InceptionResNetV2, pretrained, 512-d embeddings) |
| **Inference runtime** | TFLite (`tflite_runtime`) |
| **HLS tool** | Vitis HLS 2022.3 |
| **FPGA tool** | Vivado 2022.3 |
| **Training** | Google Colab (T4 GPU) |
| **Language** | Python 3.8+, C++ (HLS) |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | 3 persons × ~40 images |
| Embedding dimension | 512 (FaceNet) |
| Similarity threshold | 0.60 (cosine) |
| HLS conv layer II | 1 (pipelined) |
| Bitstream | ✅ Generated |

---

## 📸 Screenshots

> See Results for:
> - Vivado block design
> - PC test webcam output
> - PYNQ Jupyter notebook output

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [PYNQ Project](http://www.pynq.io/) — Xilinx Python productivity for Zynq
- [keras-facenet](https://github.com/nyoki-mtl/keras-facenet) — Pre-trained FaceNet weights
- [OpenCV](https://opencv.org/) — Haar Cascade face detection
- [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html) — HLS synthesis
- Google Colab — Free GPU for training
