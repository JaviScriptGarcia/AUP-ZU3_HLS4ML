#!/usr/bin/env python3
"""
Convert the trained MobileNet-mini (distilled_wbc.h5) to an hls4ml HLS
project for the AUP-ZU3 (xczu3eg). Mirrors the v0.2 CIFAR-10 conversion,
with a per-layer reuse_factor plan aimed at the DSP<->LUT rebalancing
discussed for v0.3.

Strategy notes (see docs/07-factores-limitantes-recursos.md):
  * The CIFAR-10 design ran at 48% LUT but only 3% DSP. At 64x64 the MACs
    roughly double, so keeping the same "everything in LUT" mapping would
    overflow the LUTs. The plan here pushes the heavy pointwise (1x1)
    convolutions onto DSPs (Resource strategy, moderate reuse) and keeps a
    *moderate* reuse -- not the lowest -- because there is a sweet spot:
    pushing reuse too low can INCREASE LUTs (the dense_resource kernel adds
    multiplexer logic). Measured in v0.2: fc1 reuse 256 -> 28685 LUT,
    reuse 128 -> 41697 LUT.
  * The stem only accepts reuse in {1,3,9,27,54,108,216,432}.

Run (host, neuralEnv10):
  cd 02.hls4ml && python convert_hls4ml.py

Output: output/myproject_wbc_prj/  (firmware + weights for the HLS stage)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.models import load_model
import hls4ml

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "01.training", "models", "distilled_wbc.h5")
OUT_DIR = os.path.join(HERE, "output", "myproject_wbc_prj")
PART = "xczu3eg-sfvc784-2-e"

# Per-layer reuse plan. Pointwise (1x1) convs are the MAC-heavy ones; give
# them a Resource strategy with moderate reuse so they map to DSPs. The
# depthwise convs are cheap (one filter per channel) -- keep them light.
# Values are starting points; the synthesis report tells us where to nudge.
REUSE_PLAN = {
    "stem":  {"Strategy": "Resource", "ReuseFactor": 27},
    "b1_pw": {"Strategy": "Resource", "ReuseFactor": 16},
    "b2_pw": {"Strategy": "Resource", "ReuseFactor": 32},
    "b3_pw": {"Strategy": "Resource", "ReuseFactor": 64},
    "b4_pw": {"Strategy": "Resource", "ReuseFactor": 64},
    "output": {"Strategy": "Resource", "ReuseFactor": 8},
}


def main():
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(MODEL, custom_objects=co)
    model.summary()

    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name",
        default_precision="fixed<16,6>", default_reuse_factor=8)

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

    # Bit-accurate check vs software, on the test set.
    te = np.load(os.path.join(HERE, "..", "00.dataset", "bloodmnist64_test.npz"))
    x_test, y_test = te["x"].astype(np.float32), te["y"]
    hls_model.compile()
    x_c = np.ascontiguousarray(x_test, dtype=np.float32)
    y_hls = np.argmax(hls_model.predict(x_c), axis=1)
    y_sw = np.argmax(model.predict(x_test, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score
    print(f"\nSoftware accuracy:           {accuracy_score(y_test, y_sw)*100:.2f}%")
    print(f"hls4ml bit-accurate accuracy: {accuracy_score(y_test, y_hls)*100:.2f}%")
    print(f"\nHLS project written to: {OUT_DIR}")
    print("Next: cd ../03.hls && vitis_hls -f build_accel.tcl")


if __name__ == "__main__":
    main()
