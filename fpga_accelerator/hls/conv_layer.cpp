#include "conv_layer.h"

/**
 * conv_layer — 2D Convolution Accelerator (HLS4FPGA)
 *
 * Implements a single convolution layer with:
 *   - K×K kernel, stride 1, valid padding
 *   - ReLU activation (sum < 0 → clamped to 0)
 *   - AXI4-Master memory interfaces for input/kernel/output
 *   - AXI4-Lite control interface (slave)
 *
 * HLS Pragmas used:
 *   - PIPELINE on innermost spatial loop → initiation interval ≈ 1
 *   - m_axi interfaces → board DDR access via AXI fabric
 *   - s_axilite → PS-side control (start/done/addresses)
 */
void conv_layer(
    data_t input [H][W][CIN],
    data_t kernel[COUT][CIN][K][K],
    data_t output[OUT_H][OUT_W][COUT]
)
{
// ── AXI Memory Interfaces (PL ↔ PS DDR) ─────────────────────────────────────
#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

// ── AXI-Lite Control Interface (ARM PS → PL) ────────────────────────────────
#pragma HLS INTERFACE s_axilite port=input   bundle=control
#pragma HLS INTERFACE s_axilite port=kernel  bundle=control
#pragma HLS INTERFACE s_axilite port=output  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

// ── Convolution Loop Nest ────────────────────────────────────────────────────
    for (int f = 0; f < COUT; f++) {          // for each output filter
        for (int i = 0; i < OUT_H; i++) {     // output row
            for (int j = 0; j < OUT_W; j++) { // output column
#pragma HLS PIPELINE                          // pipeline innermost spatial loop → II≈1

                data_t sum = 0;

                // Accumulate over input channels and kernel window
                for (int c  = 0; c  < CIN; c++ ) {
                    for (int ki = 0; ki < K;   ki++) {
                        for (int kj = 0; kj < K;   kj++) {
                            sum += input[i + ki][j + kj][c] *
                                   kernel[f][c][ki][kj];
                        }
                    }
                }

                // ReLU activation
                if (sum < 0) sum = 0;

                output[i][j][f] = sum;
            }
        }
    }
}
