#include "myproject_cifar10_accel.h"
#include "myproject.h"

// AXI-Stream + AXI-Lite wrapper around the hls4ml CIFAR-10 CNN.
//
// Protocol over AXI-Stream:
//   Input:  CIFAR10_NSAMPLES = 3072 beats of 32 bits each.
//           Each beat carries one fixed<16,6> in its low 16 bits;
//           the high 16 bits are unused. Beats are ordered as the
//           Keras model expects: row-major over (H, W, C), with
//           C running fastest (i.e. R, G, B of one pixel, then move
//           to the next pixel along the row, etc.).
//   Output: CIFAR10_NCLASSES = 10 beats of 32 bits each.
//           Each beat carries one fixed<16,6> class score (post-
//           softmax) in its low 16 bits. Last beat has TLAST = 1.
//
// The model itself is the function `myproject(...)` exported by hls4ml
// in firmware/myproject.cpp; this wrapper only does the AXI -> internal
// stream conversion on input and the internal -> AXI conversion on
// output.

void myproject_cifar10_accel(
    hls::stream<axis_t> &s_axis_in,
    hls::stream<axis_t> &s_axis_out
) {
    #pragma HLS INTERFACE axis      port=s_axis_in
    #pragma HLS INTERFACE axis      port=s_axis_out
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS DATAFLOW

    // Internal hls4ml streams. Their types come from myproject.h.
    hls::stream<input_t>  core_in("core_in");
    hls::stream<result_t> core_out("core_out");

    #pragma HLS STREAM variable=core_in  depth=32
    #pragma HLS STREAM variable=core_out depth=32

    // hls4ml represents each pixel as an `input_t` that bundles all
    // CIFAR10_CHANNELS=3 channels together (RGB in one stream element).
    // The first layer of the network (zeropad17) reads
    // CIFAR10_PIX_W * CIFAR10_PIX_H = 1024 such elements.
    //
    // Therefore the wrapper must perform exactly 1024 writes to
    // core_in, each carrying the 3 channels of a single pixel.
    // The AXI input still streams CIFAR10_NSAMPLES = 3072 beats
    // (one per channel value); they are consumed in groups of 3.
    //
    // PIPELINE II=1 goes on the INNER loop so HLS schedules one beat
    // per cycle. Total cycles: 3 * 1024 = 3072, same as before.
INPUT_LOOP:
    for (int p = 0; p < CIFAR10_PIX_W * CIFAR10_PIX_H; p++) {
        input_t pix;
    INPUT_CHANNELS:
        for (int c = 0; c < CIFAR10_CHANNELS; c++) {
            #pragma HLS PIPELINE II=1

            axis_t a = s_axis_in.read();

            ap_fixed<16, 6> fx;
            fx.range(15, 0) = a.data.range(15, 0);
            pix[c] = fx;
        }
        core_in.write(pix);
    }

    // Drive the hls4ml model.
    myproject(core_in, core_out);

    // Drain a single output vector (10 class scores).
    result_t out_vec = core_out.read();

OUTPUT_LOOP:
    for (int c = 0; c < CIFAR10_NCLASSES; c++) {
        #pragma HLS PIPELINE II=1

        axis_t ao;

        ao.data.range(15, 0)  = out_vec[c].range(15, 0);
        ao.data.range(31, 16) = 0;

        ao.keep = -1;
        ao.strb = -1;
        ao.user = 0;
        ao.id   = 0;
        ao.dest = 0;
        ao.last = (c == CIFAR10_NCLASSES - 1);

        s_axis_out.write(ao);
    }
}
