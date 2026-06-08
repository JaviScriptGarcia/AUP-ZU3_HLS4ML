# AUPZU3-HLS4ML

CNN inference acceleration on the Real Digital **AUP-ZU3** (XCZU3EG, 8 GB DDR4)
using **hls4ml** as the synthesis path and **PYNQ 3.1.1** as the runtime.

Master's thesis (TFM).

## Why hls4ml

The Vitis-AI / DPU-PYNQ path was explored first and abandoned after
confirming that **DPU-PYNQ is incompatible with PYNQ 3.1.1** (officially
confirmed by the PYNQ maintainer on the discussion forum, Nov 2025), and
that retargeting the full stack to PYNQ 3.0.1 + Vitis 2022.1 would consume
most of the available time budget.

`hls4ml` provides an alternative path that:

- Generates a pure HLS IP that is integrated into a Vivado block design and
  loaded via `pynq.Overlay()` — **no VART, no XRT, no runtime stack to
  patch**.
- Works with **Vivado 2024.1** (already installed).
- Works on the **PYNQ 3.1.1 image** already booting on the board.
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
- **PYNQ 3.1.1** — already booting on the board (validated end-to-end in the
  sibling repo `AUP-ZU3-DPU_PYNQ`, tag `v0.1-pynq-vanilla`)
- **hls4ml 1.1.0** — model-to-HLS translation (pinned by the ICTP
  environment)
- **TensorFlow 2.12 + Keras 2.12 + QKeras 0.9** — model training and
  quantization (pinned by ICTP environment)

## Upstream reference

The folder `upstream/ICTP-AUP-ZU3-HLS4ML/` is a git submodule of the ICTP
repository. It is the canonical reference for the workflow on this exact
board. The three included projects (MLP for gamma/neutron classification,
CNN for MNIST, GRU for recommendation) are followed as the starting point.

## Repository layout

```
.
├── upstream/ICTP-AUP-ZU3-HLS4ML/   git submodule of ICTP reference repo
├── docs/                            per-phase notes and decisions
├── reproduction/                    v0.1: reproduce ICTP MNIST CNN as-is
├── adaptation/                      v0.2+: own model adapted to the same flow
├── notebooks/                       Jupyter notebooks deployed to the board
├── build/                           Vivado / Vitis HLS outputs (gitignored)
└── results/                         metrics, benchmarks, comparisons
```

## Releases (git tags)

| Tag | Contents | Status |
|---|---|---|
| `v0.1-mnist-reproduction` | ICTP MNIST CNN running on our board, end-to-end | pending |
| `v0.2-own-cnn` | Own CNN trained, synthesised and running | pending |
| `v0.3-vision-model` | Larger vision CNN (target: MobileNet-style) | pending |
| `v1.0-tfm` | Thesis + benchmarks + FINN vs hls4ml comparison | pending |

## Physical location

The repository lives at `<path>`. A symlink at
`~/Documents/AUPZU3-HLS4ML` provides access from the usual path.
