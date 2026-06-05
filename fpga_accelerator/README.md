# ⚡ FPGA Accelerator — Vitis HLS + Vivado

This folder contains the first convolution layer accelerator designed for the PYNQ-Z2 board using **Vitis HLS** and **Vivado**.

## Files

| File | Description |
|------|-------------|
| `conv_layer.h` | Header — data types, dimensions, prototype |
| `conv_layer.cpp` | HLS kernel — pipelined 2D convolution + ReLU |
| `conv_layer_tb.cpp` | C testbench — uniform-input verification |

---

## Architecture

```
ARM PS (Python / TFLite)
        │
        │  AXI4-Lite (control: start / done / addresses)
        ▼
   conv_layer IP (PL)
        │
        │  AXI4-Master (data: input / kernel / output → DDR)
        ▼
    DDR Memory
```

### HLS Pragmas

| Pragma | Purpose |
|--------|---------|
| `PIPELINE` on `j` loop | Initiation interval ≈ 1 — one output pixel per cycle |
| `m_axi port=input/kernel/output bundle=gmem` | AXI4 bus to PS DDR |
| `s_axilite port=return bundle=control` | ARM-controlled start/done |

### Fixed-Point Precision

```
typedef ap_fixed<16, 8> data_t;
         ──────  ─  ─
           │     │  └── 8 fractional bits
           │     └───── 8 integer bits
           └─────────── 16 bits total
```

### Convolution Parameters (current config)

| Parameter | Value |
|-----------|-------|
| Input (H × W × CIN) | 32 × 32 × 3 |
| Kernel size | 3 × 3 |
| Output channels (COUT) | 8 |
| Stride | 1 (valid padding) |
| Output (OUT_H × OUT_W × COUT) | 30 × 30 × 8 |
| Activation | ReLU |

---

## Step 1 — C Simulation in Vitis HLS

1. Open **Vitis HLS** → New Project
2. Add source: `conv_layer.cpp`, `conv_layer.h`
3. Add testbench: `conv_layer_tb.cpp`
4. Top function: `conv_layer`
5. Part: `xc7z020clg400-1` (PYNQ-Z2)
6. Run **C Simulation** — expect:
   ```
   ✅ Simulation Done Successfully!
      Output shape : 30 x 30 x 8
      Sample output[0][0][0] = 0.269043
   ```

## Step 2 — C Synthesis

Run **C Synthesis** in Vitis HLS. Key metrics to check in the report:

| Metric | Typical result |
|--------|----------------|
| Latency (cycles) | ~7 000–10 000 |
| II (Initiation Interval) | 1 (after PIPELINE) |
| BRAM | Low (no large arrays stored on-chip) |
| DSP48 | ~9 (one per multiply-accumulate) |
| FF / LUT | Moderate |

## Step 3 — Export IP & Vivado Block Design

1. **Export RTL** from Vitis HLS → creates `conv_layer_ip.zip`
2. Open **Vivado** → New Project → PYNQ-Z2 board
3. **Create Block Design**:
   - Add **ZYNQ7 Processing System** IP
   - Add exported **conv_layer** IP
   - Connect AXI-Lite (control) and AXI-Master (gmem) via **AXI Interconnect**
   - Run **Connection Automation**
4. Validate → Generate HDL Wrapper
5. **Generate Bitstream** — takes ~10–30 min

## Step 4 — Deploy on PYNQ-Z2

After bitstream generation:

1. Copy `.bit` and `.hwh` files to `/home/xilinx/` on the board
2. Open Jupyter Notebook and run the overlay:

```python
from pynq import Overlay, allocate
import numpy as np

ol = Overlay("conv_layer.bit")
conv_ip = ol.conv_layer_0

# Allocate DMA-accessible buffers
input_buf  = allocate(shape=(32, 32, 3),     dtype=np.float32)
kernel_buf = allocate(shape=(8, 3, 3, 3),    dtype=np.float32)
output_buf = allocate(shape=(30, 30, 8),     dtype=np.float32)

# Fill with test data
input_buf[:] = 0.1
kernel_buf[:] = 0.1

# Set addresses and trigger
conv_ip.write(0x10, input_buf.physical_address)
conv_ip.write(0x18, kernel_buf.physical_address)
conv_ip.write(0x20, output_buf.physical_address)
conv_ip.write(0x00, 1)  # start

import time
while not (conv_ip.read(0x00) & 0x4):  # poll done bit
    time.sleep(0.001)

print("Output[0][0]:", output_buf[0][0])
```

---

## Screenshots

> Add Vivado block design and synthesis reports to `assets/screenshots/`:
> - `vivado_block_design.png`
> - `hls_synthesis_report.png`
> - `jupyter_pynq_output.png`

---

## Status

- [x] C simulation passing
- [x] C synthesis complete
- [x] RTL/IP export
- [x] Vivado block design
- [x] Bitstream generated
- [ ] PYNQ overlay integration (next step)
