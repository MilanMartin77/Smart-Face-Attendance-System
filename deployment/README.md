# 🚀 Deployment on PYNQ-Z2

This folder contains scripts and instructions for running the trained face recognition model on the **PYNQ-Z2** board.

## Files

| File | Description |
|------|-------------|
| `attendance_inference.py` | Main inference script — runs on PYNQ-Z2 |
| `requirements_pynq.txt` | Python dependencies for the board |

> Scripts will be added after board deployment is complete.

## Board Setup

### Prerequisites

- PYNQ-Z2 board with PYNQ v2.7+ image flashed to SD card
- USB webcam connected to the board's USB port
- Network connection (Ethernet) to access Jupyter Lab
- Trained model file (`.h5` or `.tflite`) transferred to the board

### First-Time Board Setup

1. Flash the PYNQ-Z2 SD card image from [pynq.io](http://www.pynq.io/board.html)
2. Boot the board and connect via Ethernet
3. Open Jupyter Lab at `http://192.168.2.99` (default IP)
4. Upload the model file and `attendance_inference.py` via Jupyter

### Install Dependencies on Board

```bash
# SSH into the board (or use Jupyter terminal)
pip3 install opencv-python-headless tensorflow
# or for TFLite only (lighter):
pip3 install tflite-runtime
```

### Transfer Model to Board

```bash
# From your PC (SCP)
scp face_model.h5 xilinx@192.168.2.99:/home/xilinx/
# Default password: xilinx
```

## Running Inference

```bash
# On the PYNQ-Z2 board
python3 attendance_inference.py --model /home/xilinx/face_model.h5
```

Expected output:
```
[INFO] Model loaded.
[INFO] Camera opened.
[DETECTED] Alice  — confidence: 0.97  → Attendance marked ✓
[DETECTED] Bob    — confidence: 0.91  → Attendance marked ✓
...
```

Attendance log is saved to `attendance_log.csv`:
```
Name,Timestamp,Confidence
Alice,2025-01-01 09:00:05,0.97
Bob,2025-01-01 09:00:12,0.91
```

## Notes

- The Zynq-7020 PL (FPGA fabric) is not used for inference in the current version — inference runs on the ARM PS cores
- Future work could offload convolution layers to the PL using HLS4ML or FINN
- Frame rate on the ARM cores is limited; use a lightweight model (MobileNet, SqueezeNet) for best results
