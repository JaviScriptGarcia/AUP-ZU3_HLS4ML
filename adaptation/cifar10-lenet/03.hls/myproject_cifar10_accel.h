#ifndef MYPROJECT_CIFAR10_ACCEL_H_
#define MYPROJECT_CIFAR10_ACCEL_H_

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "myproject.h"

// Wrapper that exposes the hls4ml-generated CIFAR-10 CNN with clean
// AXI-Stream input/output + AXI-Lite control. This mirrors the ICTP
// CNN/MNIST wrapper but adapted for:
//   * CIFAR-10 input: 32x32 = 1024 pixels, 3 channels (RGB) per pixel.
//     One AXI beat (32 bits) carries one fixed<16,6> sample plus zero
//     padding in the high half. We stream R, G, B as three consecutive
//     beats per pixel.
//   * Number of output classes: 10 (same as MNIST).
//   * Internal precision: fixed<16,6> to match the v3 hls4ml config.

#define CIFAR10_DATA_W       32
#define CIFAR10_PIX_W        32                 // 32x32 image side
#define CIFAR10_PIX_H        32
#define CIFAR10_CHANNELS     3                  // RGB
#define CIFAR10_NSAMPLES     (CIFAR10_PIX_W * CIFAR10_PIX_H * CIFAR10_CHANNELS) // 3072
#define CIFAR10_NCLASSES     10

typedef ap_axiu<CIFAR10_DATA_W, 1, 1, 1> axis_t;

void myproject_cifar10_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
);

#endif
