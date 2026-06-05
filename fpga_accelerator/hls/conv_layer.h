#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <ap_fixed.h>

// ── Precision ────────────────────────────────────────────────────────────────
// 16-bit fixed point: 8 integer bits, 8 fractional bits
typedef ap_fixed<16, 8> data_t;

// ── Layer Dimensions ─────────────────────────────────────────────────────────
// Input feature map
#define H    32      // height
#define W    32      // width
#define CIN  3       // input channels

// Kernel
#define K    3       // kernel size (3×3)
#define COUT 8       // output channels (number of filters)

// Output feature map dimensions (valid convolution, stride=1)
#define OUT_H  (H - K + 1)
#define OUT_W  (W - K + 1)

// ── Function Prototype ───────────────────────────────────────────────────────
void conv_layer(
    data_t input [H][W][CIN],
    data_t kernel[COUT][CIN][K][K],
    data_t output[OUT_H][OUT_W][COUT]
);

#endif // CONV_LAYER_H
