#!/usr/bin/env python3
"""
v0.5 — Convert the trained 64x64 LARGER CNN (distilled_cnn64_v05.h5) to an
hls4ml HLS project for the AUP-ZU3 (xczu3eg).

The v0.5 network is ~2x wider than v0.4: channels 32->48->96 (vs 24->32->64)
at 64x64. Conv MACs ~7.1M (1.96x v0.4's 3.61M). v0.4 fit at reuse
{27,108,144} using 157 DSP (43.6% on Vivado). To absorb ~2x the multiplies
on the SAME device we raise the reuse factors so each DSP is shared across
more MACs. The factors below are exact divisors of each layer's mult count
(Cin*k*k*Cout), as hls4ml requires:

  conv1: mult 864,   reuse 27   -> 32 active mults
  conv2: mult 13824, reuse 216  -> 64 active mults
  conv3: mult 41472, reuse 288  -> 144 active mults
  output: mult 768,  reuse 12   -> 64 active mults
  => ~304 active 24-bit mults total. At ~1-2 DSP each (24-bit > 18-bit
     native) that is ~200-330 DSP. HLS over-estimates; v0.4 HLS said
     486 DSP but Vivado gave 157. We start here and, if Vivado spills DSP
     past 360, raise conv3 reuse to 432 or 648 (both exact divisors).

Precision stays fixed<24,12>: the v0.3/v0.4 sweeps showed fixed<16,6>
collapses this network family to ~10%; <24,12> reproduced accuracy in
hardware to within ~0.2 pp.

Run (host, neuralEnv10):
  cd 02.hls4ml && PYTHONNOUSERSITE=1 python convert_cnn64_v05.py
Output: output/myproject_wbc64_v05b_prj/ (firmware + weights for the HLS stage)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.models import load_model
import hls4ml

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "01.training", "models", "distilled_cnn64_v05.h5")
OUT_DIR = os.path.join(HERE, "output", "myproject_wbc64_v05b_prj")
PART = "xczu3eg-sfvc784-2-e"
SIZE = 64

REUSE_PLAN = {
    "conv1":  {"Strategy": "Resource", "ReuseFactor": 14},
    "conv2":  {"Strategy": "Resource", "ReuseFactor": 36},
    "conv3":  {"Strategy": "Resource", "ReuseFactor": 144},
    "output": {"Strategy": "Resource", "ReuseFactor": 32},
}


def main():
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(MODEL, custom_objects=co)
    model.summary()

    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name",
        default_precision="fixed<24,12>", default_reuse_factor=8)
    cfg["Model"]["Strategy"] = "Resource"
    cfg["Model"]["IOType"] = "io_stream"
    cfg["Model"]["RoundingMode"] = "AP_RND_CONV"
    cfg["Model"]["SaturationMode"] = "AP_SAT"
    for layer, opts in REUSE_PLAN.items():
        if layer in cfg["LayerName"]:
            cfg["LayerName"][layer].update(opts)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, backend="Vitis", io_type="io_stream",
        part=PART, clock_period=10, project_name="myproject",
        output_dir=OUT_DIR)

    # Bit-accurate check vs software on the test set (native 64x64).
    te = np.load(os.path.join(HERE, "..", "00.dataset", "bloodmnist64_test.npz"))
    x_test = te["x"].astype(np.float32)   # already 64x64, no resize
    y_test = te["y"]
    hls_model.compile()
    x_c = np.ascontiguousarray(x_test, dtype=np.float32)
    y_hls = np.argmax(hls_model.predict(x_c), axis=1)
    y_sw = np.argmax(model.predict(x_test, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score
    print(f"\nSoftware accuracy:            {accuracy_score(y_test, y_sw)*100:.2f}%")
    print(f"hls4ml bit-accurate accuracy: {accuracy_score(y_test, y_hls)*100:.2f}%")
    print(f"(v0.4 reference was 89.16% sw / 89.04% bit-accurate at 64x64)")

    # Save per-image hls4ml predictions for the 1:1 hardware comparison.
    np.save(os.path.join(HERE, "ref_pred_sw64_v05b.npy"), y_sw)
    print(f"\nHLS project written to: {OUT_DIR}")
    print("Next: cd ../03.hls && build accel (12288 beats), then vitis_hls -f build_accel.tcl")


if __name__ == "__main__":
    main()
