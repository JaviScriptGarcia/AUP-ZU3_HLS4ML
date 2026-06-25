#include "myproject_wbc_accel.h"
#include "myproject.h"

// AXI-Stream + AXI-Lite wrapper around the hls4ml MobileNet-mini for
// BloodMNIST. See the header for the protocol.
//
// Input:  WBC_NSAMPLES = 12288 beats of 32 bits each. Each beat carries
//         one fixed<16,6> in its low 16 bits. Beats are ordered row-major
//         over (H, W, C) with C running fastest (R, G, B of one pixel,
//         then the next pixel along the row).
// Output: WBC_NCLASSES = 8 beats, one fixed<16,6> class score each.
//         Last beat has TLAST = 1.

void myproject_wbc_accel(
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

    #pragma HLS STREAM variable=core_in  depth=64
    #pragma HLS STREAM variable=core_out depth=8

    // hls4ml bundles the WBC_CHANNELS=3 RGB channels into one input_t
    // element. The network reads WBC_PIX_W * WBC_PIX_H = 4096 such
    // elements, so the wrapper performs exactly 4096 writes to core_in,
    // each carrying the 3 channels of a single pixel. The AXI input
    // streams WBC_NSAMPLES = 12288 beats, consumed in groups of 3.
    //
    // PIPELINE II=1 on the INNER loop: one beat per cycle, 3 * 4096 cycles.
INPUT_LOOP:
    for (int p = 0; p < WBC_PIX_W * WBC_PIX_H; p++) {
        input_t pix;
    INPUT_CHANNELS:
        for (int c = 0; c < WBC_CHANNELS; c++) {
            #pragma HLS PIPELINE II=1

            axis_t a = s_axis_in.read();

            // Network input is ap_fixed<24,12>; the AXI beat carries the
            // 24-bit value in its low 24 bits (high 8 unused).
            ap_fixed<24, 12> fx;
            fx.range(23, 0) = a.data.range(23, 0);
            pix[c] = fx;
        }
        core_in.write(pix);
    }

    // Drive the hls4ml model.
    myproject(core_in, core_out);

    // Drain a single output vector (8 class scores).
    result_t out_vec = core_out.read();

OUTPUT_LOOP:
    for (int c = 0; c < WBC_NCLASSES; c++) {
        #pragma HLS PIPELINE II=1

        axis_t ao;

        // Output scores are ap_fixed<24,12>: pack 24 bits, zero the rest.
        ao.data.range(23, 0)  = out_vec[c].range(23, 0);
        ao.data.range(31, 24) = 0;

        ao.keep = -1;
        ao.strb = -1;
        ao.user = 0;
        ao.id   = 0;
        ao.dest = 0;
        ao.last = (c == WBC_NCLASSES - 1);

        s_axis_out.write(ao);
    }
}
