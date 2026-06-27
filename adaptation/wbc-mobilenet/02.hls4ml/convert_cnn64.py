#!/usr/bin/env python3
"""
v0.4 — Convert the trained 64x64 wider CNN (distilled_cnn64.h5) to an
hls4ml HLS project for the AUP-ZU3 (xczu3eg).

The v0.4 network has ~3.2x more conv MACs than v0.3 (3.61M vs 1.12M):
channels 24->32->64 at 64x64 vs 16->24->48 at 48x48. v0.3 already used
341 DSP (95% of 360) at reuse {9,36,48}. To fit the bigger network in the
SAME device we raise the reuse factors ~3-4x (each DSP is shared across
more MACs over more cycles), trading latency for area. The reuse factors
below are exact divisors of each layer's mult count (Cin*k*k*Cout), as
hls4ml requires, chosen conservatively to leave DSP headroom:

  conv1: mult 648,   reuse 27   -> ~24 active mults
  conv2: mult 6912,  reuse 144  -> ~48 active mults
  conv3: mult 18432, reuse 192  -> ~96 active mults
  => ~168 active 24-bit mults; at 1-2 DSP each that is ~168-336 DSP,
     comfortably under 360. Latency rises ~3x (still << 2 ms/image).

Precision stays fixed<24,12>: the v0.3 sweep showed fixed<16,6> collapses
this family of networks to ~10%; <24,12> reproduced 83% in hardware to
within 0.14 pp. Same reasoning applies here.

Run (host, neuralEnv10):
  cd 02.hls4ml && PYTHONNOUSERSITE=1 python convert_cnn64.py
Output: output/myproject_wbc64_prj/  (firmware + weights for the HLS stage)
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
MODEL = os.path.join(HERE, "..", "01.training", "models", "distilled_cnn64.h5")
OUT_DIR = os.path.join(HERE, "output", "myproject_wbc64_prj")
PART = "xczu3eg-sfvc784-2-e"
SIZE = 64

REUSE_PLAN = {
    "conv1":  {"Strategy": "Resource", "ReuseFactor": 27},
    "conv2":  {"Strategy": "Resource", "ReuseFactor": 144},
    "conv3":  {"Strategy": "Resource", "ReuseFactor": 192},
    "output": {"Strategy": "Resource", "ReuseFactor": 8},
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
    print(f"(v0.3 reference was 83.05% sw / 83.13% bit-accurate at 48x48)")

    # Save per-image hls4ml predictions for the 1:1 hardware comparison.
    np.save(os.path.join(HERE, "ref_pred_sw64.npy"), y_sw)
    print(f"\nHLS project written to: {OUT_DIR}")
    print("Next: cd ../03.hls && update wrapper to 12288 beats, then vitis_hls -f build_accel.tcl")


if __name__ == "__main__":
    main()
