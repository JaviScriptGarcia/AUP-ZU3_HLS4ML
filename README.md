# AUPZU3-HLS4ML

CNN inference acceleration on the Real Digital **AUP-ZU3** (XCZU3EG, 8 GB DDR4)
using **hls4ml** as the synthesis path and **PYNQ 3.1.1** as the runtime.

## Why hls4ml

`hls4ml` generates a pure HLS IP that is integrated into a Vivado block
design and loaded with `pynq.Overlay()`:

- No VART, no XRT — no separate runtime stack to install or patch.
- Works with **Vivado 2024.1** and the **PYNQ 3.1.1** image on the board.
- Has direct upstream support for the AUP-ZU3 board via the
  [ICTP/AUP-ZU3-HLS4ML](https://github.com/ICTP/AUP-ZU3-HLS4ML) reference
  repo from the AMD University Program.

## Hardware

- Real Digital AUP-ZU3 (8 GB variant)
- ZynqMP XCZU3EG-SFVC784-2-e
- 8 GB DDR4-2400T on 32-bit bus
- microSD socket boot, PYNQ 3.1.1 image

## Software / toolchain

- **Vivado 2024.1** + **Vitis HLS 2024.1** — block design and IP synthesis
- **PYNQ 3.1.1** — runtime on the board
- **hls4ml 1.1.0** — model-to-HLS translation (pinned by the ICTP
  environment)
- **TensorFlow 2.12 + Keras 2.12 + QKeras 0.9** — model training and
  quantization (pinned by the ICTP environment)

## Upstream reference

The folder `upstream/ICTP-AUP-ZU3-HLS4ML/` is a git submodule of the ICTP
repository, the reference for the workflow on this board. The three included
projects (MLP for gamma/neutron classification, CNN for MNIST, GRU for
recommendation) are the starting point.

## Repository layout

```
.
├── upstream/ICTP-AUP-ZU3-HLS4ML/   git submodule of ICTP reference repo
├── reproduction/                    reproduce the ICTP MNIST CNN as-is
├── adaptation/                      own model adapted to the same flow
├── notebooks/                       Jupyter notebooks deployed to the board
├── build/                           Vivado / Vitis HLS outputs (gitignored)
└── results/                         metrics and comparisons
```

## Experiments

Each row is an end-to-end result: trained model → hls4ml → Vivado bitstream →
PYNQ overlay → inference measured **on the board**. Latency and throughput are
the hardware-only figures from the deployed inference notebooks.

| # | Experiment | HW accuracy | HW latency / image | Throughput |
|---|---|---|---|---|
| v0.1 | MNIST CNN reproduced end-to-end (ICTP reference) | 90.54 % | — | — |
| v0.2 | Own CIFAR-10 CNN trained, synthesised and run | 60.90 % (60.97 % bit-accurate) | — | — |
| v0.3 | BloodMNIST white-blood-cell CNN, 48×48 | 83.19 % | — | — |
| v0.4 | BloodMNIST CNN widened to 64×64 (24→32→64 ch) | 88.98 % | 721 µs | 1387 img/s |
| **v0.5** | **BloodMNIST 64×64, wider net (32→48→96 ch) + KD + hyperparameter search** | **93.19 %** | **739 µs** | **1353 img/s** |

**v0.5 is the current deployed design.** The quantised `ap_fixed<24,12>`
network reproduces the software accuracy in hardware to within 0.06 pp
(93.25 % software → 93.19 % on the board, 99.42 % per-image agreement). It uses
roughly half the device on the XCZU3EG (Vivado: LUT 50.5 %, DSP 43.6 %,
BRAM 57.4 %), with timing met (WNS +0.613 ns at 100 MHz).

### Notes on the design space

- **Resolution matters, not just upscaling.** BloodMNIST 64×64 has real
  microscopy detail; PathMNIST/DermaMNIST 64×64 were found to be interpolated
  from 28×28 (no high-frequency gain), so they were not used.
- **Reuse-factor co-design.** Lowering the conv reuse factors cuts latency
  (a v0.5b variant reached 634 µs in HLS, ~2.6× faster) but pushes the conv
  multiplies from DSP into LUTs; on this device that overflowed the LUT budget
  (Vivado placement: 104 % LUT), so the higher-reuse v0.5 remains the deployed
  design. This is a concrete LUT-vs-latency limit of the XCZU3EG for this net.

See `adaptation/cifar10-lenet/README.md` for the CIFAR-10 reproduction and
`adaptation/wbc-mobilenet/` for the BloodMNIST experiments.
