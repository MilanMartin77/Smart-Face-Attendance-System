#include <iostream>
#include <cmath>
#include "conv_layer.h"

/**
 * Testbench for conv_layer HLS kernel.
 *
 * Strategy:
 *   - Fill input with uniform value 0.1
 *   - Fill all kernel weights with uniform value 0.1
 *   - Expected output for each output pixel:
 *       sum = CIN * K * K * (0.1 * 0.1) = 3 * 3 * 3 * 0.01 = 0.27
 *   - Since sum > 0, ReLU leaves it unchanged.
 *
 * Run in Vitis HLS:  Project → Run C Simulation
 */

static data_t input [H][W][CIN];
static data_t kernel[COUT][CIN][K][K];
static data_t output[OUT_H][OUT_W][COUT];

int main()
{
    // ── Initialise inputs ─────────────────────────────────────────────────────
    for (int i = 0; i < H;    i++)
    for (int j = 0; j < W;    j++)
    for (int c = 0; c < CIN;  c++)
        input[i][j][c] = (data_t)0.1;

    for (int f  = 0; f  < COUT; f++ )
    for (int c  = 0; c  < CIN;  c++ )
    for (int ki = 0; ki < K;    ki++)
    for (int kj = 0; kj < K;    kj++)
        kernel[f][c][ki][kj] = (data_t)0.1;

    // ── Run HLS kernel ────────────────────────────────────────────────────────
    conv_layer(input, kernel, output);

    // ── Verify ───────────────────────────────────────────────────────────────
    // Expected: CIN * K * K * 0.1 * 0.1 = 3*3*3*0.01 = 0.27 for every output cell
    const float EXPECTED = (float)(CIN * K * K) * 0.01f;
    const float TOLERANCE = 0.01f;   // fixed-point rounding allowance

    int errors = 0;
    for (int f = 0; f < COUT;  f++)
    for (int i = 0; i < OUT_H; i++)
    for (int j = 0; j < OUT_W; j++) {
        float val = (float)output[i][j][f];
        if (std::fabs(val - EXPECTED) > TOLERANCE) {
            std::cerr << "MISMATCH at [" << i << "][" << j << "][" << f << "] "
                      << "got=" << val << " expected=" << EXPECTED << std::endl;
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "✅ Simulation Done Successfully!" << std::endl;
        std::cout << "   Output shape : " << OUT_H << " x " << OUT_W
                  << " x " << COUT << std::endl;
        std::cout << "   Sample output[0][0][0] = "
                  << (float)output[0][0][0] << std::endl;
        return 0;
    } else {
        std::cout << "❌ " << errors << " mismatches found!" << std::endl;
        return 1;
    }
}
