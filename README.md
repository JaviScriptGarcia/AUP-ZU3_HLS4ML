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

| Experiment | Result |
|---|---|
| MNIST CNN reproduced end-to-end on the AUP-ZU3 | 90.54 % hardware accuracy |
| Own CIFAR-10 CNN trained, synthesised and run on the FPGA | 60.90 % hardware accuracy (matches the 60.97 % bit-accurate simulation) |

See `adaptation/cifar10-lenet/README.md` for the full CIFAR-10 reproduction
steps (board-only inference path and full-rebuild path).
