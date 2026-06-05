# 📊 Results & Evaluation

## Dataset

| Person | Images Collected | Gallery strategy |
|--------|-----------------|-----------------|
| Member 1 | ~40 | Centroid of all embeddings |
| Member 2 | ~40 | Centroid of all embeddings |
| Member 3 | ~40 | Centroid of all embeddings |

## Training (Google Colab)

| Metric | Value |
|--------|-------|
| Model | FaceNet (InceptionResNetV2, pretrained) |
| Embedding dim | 512 |
| Similarity metric | Cosine similarity |
| float32 TFLite | ✅ Generated |
| int8 TFLite | ✅ Generated |

## PC Test Results

| Metric | Value |
|--------|-------|
| Similarity threshold | 0.60 |
| Required frames for PRESENT | 15 |
| Recognition accuracy | _(add after testing)_ |

## PYNQ-Z2 Deployment

| Metric | Value |
|--------|-------|
| Inference FPS | _(add after testing)_ |
| Avg. confidence score | _(add after testing)_ |

## FPGA Accelerator (Vitis HLS + Vivado)

| Metric | Value |
|--------|-------|
| Kernel | 2D Conv + ReLU |
| Input | 32 × 32 × 3, Kernel 3×3, 8 filters |
| Output | 30 × 30 × 8 |
| Precision | ap_fixed<16,8> |
| Pipeline II | 1 |
| C simulation | ✅ Passed |
| C synthesis | ✅ Complete |
| Vivado block design | ✅ Complete |
| Bitstream | ✅ Generated |
