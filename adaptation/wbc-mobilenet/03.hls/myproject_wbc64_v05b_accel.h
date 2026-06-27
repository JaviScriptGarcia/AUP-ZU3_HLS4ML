#ifndef MYPROJECT_WBC64_V05B_ACCEL_H_
#define MYPROJECT_WBC64_V05B_ACCEL_H_

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "myproject.h"

// AXI-Stream + AXI-Lite wrapper around the hls4ml-generated standard CNN
// for BloodMNIST (white blood cell classification), 64x64 input (v0.5b).
// The external interface is IDENTICAL to v0.4 — only the internal hls4ml
// core is wider (channels 32->48->96 vs 24->32->64), giving 93.25% software
// accuracy. Interface:
//   * Input: 64x64 = 4096 pixels, 3 channels (RGB) per pixel.
//     One AXI beat (32 bits) carries one fixed<24,12> sample in its low
//     24 bits. R, G, B are streamed as three consecutive beats per pixel,
//     so 4096 * 3 = 12288 beats per image.
//   * Output: 8 classes (BloodMNIST cell types).
//   * Internal precision: fixed<24,12>, matching the hls4ml config.

#define WBC_DATA_W       32
#define WBC_PIX_W        64                          // 64x64 image side
#define WBC_PIX_H        64
#define WBC_CHANNELS     3                           // RGB
#define WBC_NSAMPLES     (WBC_PIX_W * WBC_PIX_H * WBC_CHANNELS) // 12288
#define WBC_NCLASSES     8

typedef ap_axiu<WBC_DATA_W, 1, 1, 1> axis_t;

void myproject_wbc64_v05b_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);

#endif
