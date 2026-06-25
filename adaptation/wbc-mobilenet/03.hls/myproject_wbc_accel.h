#ifndef MYPROJECT_WBC_ACCEL_H_
#define MYPROJECT_WBC_ACCEL_H_

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "myproject.h"

// AXI-Stream + AXI-Lite wrapper around the hls4ml-generated standard CNN
// for BloodMNIST (white blood cell classification), 48x48 input.
//   * Input: 48x48 = 2304 pixels, 3 channels (RGB) per pixel.
//     One AXI beat (32 bits) carries one fixed<16,6> sample in its low
//     16 bits. R, G, B are streamed as three consecutive beats per pixel,
//     so 2304 * 3 = 6912 beats per image.
//   * Output: 8 classes (BloodMNIST cell types).
//   * Internal precision: fixed<16,6>, matching the hls4ml config.

#define WBC_DATA_W       32
#define WBC_PIX_W        48                          // 48x48 image side
#define WBC_PIX_H        48
#define WBC_CHANNELS     3                           // RGB
#define WBC_NSAMPLES     (WBC_PIX_W * WBC_PIX_H * WBC_CHANNELS) // 6912
#define WBC_NCLASSES     8

typedef ap_axiu<WBC_DATA_W, 1, 1, 1> axis_t;

void myproject_wbc_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);

#endif
