#!/usr/bin/env python3
"""
Convert the trained 48x48 standard CNN (distilled_cnn48.h5) to an hls4ml
HLS project for the AUP-ZU3 (xczu3eg).

Reuse plan: the three convolutions are the MAC-heavy layers. Use the
Resource strategy with a moderate reuse factor so multiplications map to
DSPs (plenty free) rather than LUTs (the scarce resource). A moderate
reuse, not the lowest, because pushing reuse too low can increase LUTs
(measured in v0.2). The final dense head is tiny.

Run (host, neuralEnv10):
  cd 02.hls4ml && python convert_cnn48.py
Output: output/myproject_wbc_prj/  (firmware + weights for the HLS stage)
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
MODEL = os.path.join(HERE, "..", "01.training", "models", "distilled_cnn48.h5")
OUT_DIR = os.path.join(HERE, "output", "myproject_wbc_prj")
PART = "xczu3eg-sfvc784-2-e"
SIZE = 48

REUSE_PLAN = {
    "conv1":  {"Strategy": "Resource", "ReuseFactor": 9},
    "conv2":  {"Strategy": "Resource", "ReuseFactor": 36},
    "conv3":  {"Strategy": "Resource", "ReuseFactor": 48},
    "output": {"Strategy": "Resource", "ReuseFactor": 8},
}


def main():
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(MODEL, custom_objects=co)
    model.summary()

    # This network needs more dynamic range than the CIFAR design: a
    # precision sweep showed fixed<16,6> collapses it to ~10% (random)
    # while fixed<24,12> reproduces the 83% software accuracy. The wider
    # word costs more DSP/LUT but is required for the model to work.
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

    # Bit-accurate check vs software on the test set (resized to 48).
    te = np.load(os.path.join(HERE, "..", "00.dataset", "bloodmnist64_test.npz"))
    x_test = tf.image.resize(te["x"], (SIZE, SIZE)).numpy().astype(np.float32)
    y_test = te["y"]
    hls_model.compile()
    x_c = np.ascontiguousarray(x_test, dtype=np.float32)
    y_hls = np.argmax(hls_model.predict(x_c), axis=1)
    y_sw = np.argmax(model.predict(x_test, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score
    print(f"\nSoftware accuracy:            {accuracy_score(y_test, y_sw)*100:.2f}%")
    print(f"hls4ml bit-accurate accuracy: {accuracy_score(y_test, y_hls)*100:.2f}%")
    print(f"\nHLS project written to: {OUT_DIR}")
    print("Next: cd ../03.hls && vitis_hls -f build_accel.tcl")


if __name__ == "__main__":
    main()
