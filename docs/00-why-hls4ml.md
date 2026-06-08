# Why hls4ml (and not DPU-PYNQ)

## What was tried before

The sibling repo `AUPZU3-DPU_PYNQ` explored the path of running a Xilinx
DPU on the AUP-ZU3 via the DPU-PYNQ runtime. The exploration reached:

- `v0.1-pynq-vanilla`: PYNQ 3.1.1 booting on the board, base overlay
  loading, SSH and Jupyter reachable.
- Attempt to install `pip3 install pynq-dpu` — succeeded but `load_model()`
  silently crashed the Python kernel.
- Root cause confirmed: VART 2.5 (installed by the PYNQ image) is linked
  against an old XRT ABI; the XRT 2.17 that ships in PYNQ 3.1.1 dropped
  symbols (`xclClose`) that VART still calls.
- Patch attempt with the official AMD Kria zip
  (`vai3.5_kr260.zip`) upgraded VART to 3.5 and Boost to 1.80, but
  `xclClose` still missing → dead end.

The official answer from the PYNQ maintainer
(joshgoldsmith on the PYNQ forum, Nov 2025) is unequivocal:

> *"DPU-PYNQ is not supported in PYNQ v3.1 due to vart compatibility.
> PYNQ v3.0.1 still supports it though, so I would suggest porting your
> board to that version instead."*

Porting to PYNQ 3.0.1 would require:

- Downloading and installing Vivado 2022.1, Vitis 2022.1 and PetaLinux
  2022.1 (~100 GB).
- Rebuilding the AUP-ZU3 image with the 2022.1 toolchain.
- Adapting the AUP-ZU3 `gen_platform.tcl` for the older toolchain.
- Reflashing the SD card and starting over.

Estimated cost: 10-15 days with non-trivial risk. With ~27 days of TFM
deadline left, that consumed too much of the budget.

## Why hls4ml is a better fit here

`hls4ml` is a different abstraction. Instead of compiling a generic
quantised model to a fixed DPU IP, it translates a specific trained model
into a custom HLS C++ pipeline that becomes its own IP. That IP is then
dropped into a Vivado block design, packaged into a bitstream, and loaded
via `pynq.Overlay()` like any other PL overlay.

Consequences relevant to this project:

| Property | DPU-PYNQ | hls4ml |
|---|---|---|
| Runtime on board | VART + XIR + libvart-* + XRT | none — pure `pynq.Overlay` + DMA |
| PYNQ 3.1.1 support | broken | works (no extra runtime to mismatch) |
| Vivado version | 2022.1 (mandatory) | 2024.1 (already installed) |
| Per-model bitstream | no, one DPU runs any xmodel | yes, every model needs synthesis |
| Resource cost | DPU is fixed ~70-80% of ZU3 PL | depends on model / quantisation / reuse factor |
| AUP-ZU3 upstream | not supported | supported by ICTP reference repo |

The "per-model bitstream" property is a real cost: changing the model
means re-synthesising the FPGA. For a thesis with a fixed model target,
that cost is acceptable; it is paid once and re-used.

## The ICTP reference

The [ICTP/AUP-ZU3-HLS4ML](https://github.com/ICTP/AUP-ZU3-HLS4ML) repo,
sponsored by the AMD University Program, ships:

- Three end-to-end examples (MLP, CNN, GRU) for the exact AUP-ZU3 board
- A wiki with the full workflow (training → compression → hls4ml → Vivado
  → PYNQ)
- Pre-generated `.xsa` files for the AUP-ZU3 board
- A `cnn_bd.tcl` reproducing the Vivado block design from scratch
- An AXI4-Stream + AXI-Lite + DMA template that wraps any hls4ml output

We follow this as the upstream and adapt it to our own model.

## Constraint on model size

The ZU3EG has 154K LCs, 360 DSPs, 7.6 Mb URAM and 1.8 Mb BRAM. This is
small for full vision models. hls4ml resource use depends on:

- The model itself (channels, depth)
- Quantisation precision (the smaller, the cheaper)
- The `reuse_factor` parameter (which trades latency for resource sharing)
- Whether layers stay on-chip or stream through DRAM

A direct MobileNet implementation is unlikely to fit. The practical target
is a smaller vision CNN (MobileNet-lite, MicroNet-style) trained from
scratch with QKeras, with `reuse_factor` tuned to keep DSP and BRAM under
the device's limits.
