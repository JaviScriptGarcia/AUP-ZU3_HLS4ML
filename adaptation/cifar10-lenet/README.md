# CIFAR-10 CNN on AUP-ZU3 (hls4ml → PYNQ)

A small quantised convolutional neural network trained on CIFAR-10,
compiled to FPGA logic with [hls4ml](https://fastmachinelearning.org/hls4ml/),
and run on the Real Digital **AUP-ZU3** board (AMD Zynq UltraScale+
XCZU3EG) through the PYNQ framework.

This follows the end-to-end workflow of the
[ICTP/AUP-ZU3-HLS4ML](https://github.com/ICTP/AUP-ZU3-HLS4ML) reference
(training → compression → hls4ml → AXI wrapper → Vivado → PYNQ), but
with our own model on a harder dataset: CIFAR-10 (32×32 RGB, 10 object
classes) instead of MNIST.

## Result

| Stage | Accuracy on the 10 000-sample CIFAR-10 test set |
|---|---|
| Software (QKeras, float)            | 61.41 % |
| hls4ml bit-accurate simulation      | 60.97 % |
| **Hardware (FPGA, measured)**       | **60.90 %** |

The hardware reproduces the bit-accurate simulation within 0.07
percentage points. Throughput is ~1190 inferences/s (bound by the
Python loop; the pure HLS latency is ~78 µs/image). The model is a
~14 K-parameter distilled student and uses `ap_fixed<16,6>` precision.

---

## Two ways to use this repo

### Path A — Reproduce inference on the board (NO Vivado needed)

This is the fast path. It needs **only an AUP-ZU3 board running the
official PYNQ 3.1.1 image** — no Vivado, no Vitis HLS, no big RAM.
The synthesised overlay is already committed as
`04.hw/bd_wrapper_cifar10.xsa` (3.8 MB). A `.xsa` *is* a PYNQ overlay:
it bundles the bitstream (`.bit`) and the hardware handoff (`.hwh`),
and `pynq.Overlay()` loads it directly.

**Steps:**

1. On the host, generate the test CSV (the dataset itself is not
   committed because it is 559 MB). In an environment with TensorFlow
   (the `neuralEnv10` conda env, see Path B step 1):

   ```python
   import os, numpy as np, pandas as pd
   from tensorflow.keras.datasets import cifar10
   (_, _), (x_test, y_test) = cifar10.load_data()
   x = (x_test.astype(np.float32) / 255.0).reshape(len(x_test), -1)
   y = y_test.flatten().astype(int)[:, None]
   out = np.hstack([x, y])
   cols = [f'p{i}' for i in range(x.shape[1])] + ['label']
   os.makedirs('test_dataset', exist_ok=True)
   pd.DataFrame(out, columns=cols).to_csv('test_dataset/cifar10_test.csv', index=False)
   ```

   (If you have no TensorFlow handy, any source of the CIFAR-10 test
   set works — the CSV just needs 3072 normalised float pixel columns
   in HWC order with the channel running fastest, plus a final integer
   `label` column.)

2. Copy four things to a folder on the board (e.g.
   `/home/xilinx/jupyter_notebooks/cifar10/`):

   ```bash
   scp 04.hw/bd_wrapper_cifar10.xsa      xilinx@<board-ip>:/home/xilinx/jupyter_notebooks/cifar10/
   scp 05.pynq/02-cifar10.ipynb          xilinx@<board-ip>:/home/xilinx/jupyter_notebooks/cifar10/
   scp 05.pynq/utils.py                  xilinx@<board-ip>:/home/xilinx/jupyter_notebooks/cifar10/
   ssh xilinx@<board-ip> 'mkdir -p /home/xilinx/jupyter_notebooks/cifar10/test_dataset'
   scp test_dataset/cifar10_test.csv     xilinx@<board-ip>:/home/xilinx/jupyter_notebooks/cifar10/test_dataset/
   ```

3. Open `02-cifar10.ipynb` in the board's Jupyter
   (`http://<board-ip>:9090`) and run the cells top to bottom. The
   final cell reports the hardware accuracy over the full test set.

That's it. No FPGA tools required on your machine.

### Path B — Regenerate everything from scratch (needs Vivado)

Only do this if you want to retrain the model, change the
architecture, or rebuild the bitstream. It uses the full toolchain.

> ⚠️ **Hard requirement: ~32 GB of RAM to synthesise.**
> The Vivado synthesis of the complete block design (ZynqMP PS + AXI
> DMA + interconnects + the hls4ml IP) peaks at ~25.6 GB of resident
> memory. On a 15 GB host it swap-thrashes and the OS freezes. If you
> cannot provide ~32 GB, **use Path A instead**; the committed XSA
> already contains a working bitstream.

Tooling used (other versions may behave differently):
- Vivado / Vitis HLS **2024.1**
- AUP-ZU3 board files from
  [RealDigitalOrg/aup-zu3-bsp](https://github.com/RealDigitalOrg/aup-zu3-bsp/tree/master/board-files),
  copied into `<Vivado>/data/boards/board_files/`
- Python env `neuralEnv10` (conda): `tensorflow==2.12.0`,
  `keras==2.12.0`, `QKeras==0.9.0`, `hls4ml==1.1.0`
  (see the ICTP repo's `environment/` for the full spec)

Stage order:

| Stage | Folder | What it does | Output |
|---|---|---|---|
| 1 | `01.training/` | Train teacher + distil student (QKeras) | `models/distilled_cifar10.h5` |
| 2 | `02.hls4ml/`   | Convert to Vitis HLS C++ (`fixed<16,6>`) | `output/.../firmware/` |
| 3 | `03.hls/`      | Wrap in AXI-Stream + AXI-Lite, synthesise IP | packaged IP |
| 4 | `04.hw/`       | Vivado block design, synth, impl, bitstream | `bd_wrapper_cifar10.xsa` |
| 5 | `05.pynq/`     | Run inference on the board | accuracy |

Build commands for stages 3–4 (host with Vivado 2024.1 sourced):

```bash
# Stage 3 — synthesise the AXI wrapper IP (~5 min, light on RAM)
cd 03.hls
vitis_hls -f build_accel.tcl

# Stage 4 — block design + synth + impl + bitstream + XSA (~50 min, 32 GB RAM)
cd ../04.hw
vivado -mode batch -source build_bitstream.tcl
```

Then follow Path A from step 1 to deploy the freshly built XSA.

---

## What's in this folder

```
cifar10-lenet/
├── 01.training/cifar10-training.ipynb   training + knowledge distillation
├── 02.hls4ml/cifar10-hls4ml.ipynb       hls4ml conversion + IP export
├── 03.hls/                              AXI wrapper around the hls4ml IP
│   ├── myproject_cifar10_accel.cpp      (groups 3 RGB beats → 1 pixel)
│   ├── myproject_cifar10_accel.h
│   └── build_accel.tcl
├── 04.hw/                               Vivado hardware build
│   ├── cifar10_bd.tcl                   block design (PS + DMA + IP)
│   ├── build_bitstream.tcl              synth → impl → bitstream → XSA
│   └── bd_wrapper_cifar10.xsa           ← the committed overlay (Path A)
└── 05.pynq/                             on-board inference
    ├── 02-cifar10.ipynb
    └── utils.py
```

Regenerable artefacts (HLS/Vivado project trees, the 559 MB test CSV,
the trained `.h5`) are git-ignored; the notebooks and TCL scripts
regenerate them.

## Notes / gotchas worth knowing

- **AXI wrapper pixel packing.** hls4ml types the network input as
  `array<fixed<16,6>, 3>` — all 3 RGB channels of one pixel bundled
  into one stream element. The first layer (zeropad, 32×32) consumes
  1024 such elements. The wrapper therefore reads 3 AXI beats (R, G, B)
  per pixel and writes once to the internal stream → 1024 writes, not
  3072. Getting this wrong over-fills the depth-32 FIFO and
  back-pressures the DMA into a deterministic hang.
- **PYNQ 3.1.1 has no TensorFlow.** Generate the test CSV on the host,
  not on the board.
- **Fixed-point precision matters.** `ap_fixed<16,12>` (the ICTP MNIST
  default) collapses this model to ~10 % (random) because it leaves
  only 4 fractional bits. `ap_fixed<16,6>` (10 fractional bits) keeps
  it at ~61 %.
