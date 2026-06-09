# Journey summary — from FINN to DPU to hls4ml

This document recaps the full path of the TFM exploration, from the initial
FINN+PYNQ prototype through the DPU-PYNQ attempt to the current hls4ml
implementation. It is intended both as a record of decisions for the
thesis chapter on critical analysis of FPGA AI deployment paths, and as
context for anyone reviewing the repository.

---

## Stage 1 — FINN + PYNQ (prior work, separate conversation thread)

### What was done

A first attempt at running a CNN on the AUP-ZU3 was made using the
[FINN compiler](https://github.com/Xilinx/finn) developed by Xilinx
Research. FINN targets extremely quantised neural networks (1-bit, 2-bit
weights and activations) and synthesises them as fully dataflow
streaming architectures.

The output was a working accelerator on the AUP-ZU3 running a small
classifier with binarised weights and activations.

### What FINN does in technical terms

- **Quantisation**: aggressive. 1-bit weights (binary) and 1-bit
  activations (signs only). Optionally 2-bit or 4-bit.
- **Compute style**: dataflow. Each layer is a hardware block with its
  own pipeline; data flows from one to the next via streams.
- **Resource usage**: 1-bit multiplications collapse to XNOR gates and
  popcount. Almost no DSP usage, mostly LUTs.
- **Deployment**: pure `pynq.Overlay()` + DMA. No runtime stack.

### Limitations observed

- Accuracy heavily impacted by 1-bit quantisation. Useful for toy
  datasets (binarised MNIST or small CIFAR variants) but quickly
  degrades on richer vision tasks.
- Building blocks for popular vision architectures (MobileNet's
  depthwise-separable convolutions, residual blocks) are not first-class
  in FINN's library and require non-trivial work to integrate.
- The artefact felt more like "a binarised CNN demo" than a generalist
  vision inference pipeline.

### Why this was not enough for the thesis goal

The TFM's objective is to run a recognisable vision model (MobileNet-class
or similar) on the AUP-ZU3 with reasonable accuracy. FINN's 1-bit world
makes that very hard. Hence the search for an alternative path with
richer numerics (INT8 or higher fixed-point).

---

## Stage 2 — PetaLinux + Hackster DPU attempt (failed silently)

### What was done

Following the [Hackster.io tutorial for MYD-CZU3EG with DPUCZDX8G v4.1
B1600](https://www.hackster.io/whitney-knitter/...), a PetaLinux project
was built with:

- PetaLinux 2024.1 from the official AUP-ZU3 BSP.
- The Hackster TCL describing a Block Design with PSU + clk_wiz + DPU
  + interconnects + xlconcat.
- meta-vitis layer integrated for the VART runtime.
- A custom BOOT.BIN flashed to SD.

### What happened

The board powered on, the PS LEDs blinked briefly, and the UART stayed
**silent forever**. No FSBL banner, no U-Boot, no kernel. Dead boot.

### Root cause (found by analysing the TCLs side by side)

The Hackster TCL was written for the **MYD-CZU3EG** board, which shares
the XCZU3EG silicon but **has different PCB wiring**:

| Parameter | Hackster TCL (MYD-CZU3EG) | Real AUP-ZU3 8GB |
|---|---|---|
| `PSU__DDRC__BUS_WIDTH` | 64 Bit | 32 Bit |
| `PSU__DDRC__DRAM_WIDTH` | 16 Bits | 8 Bits |
| `PSU__DDRC__SPEED_BIN` | DDR4_2400R | DDR4_2400T |
| `PSU__DDRC__DEVICE_CAPACITY` | 8192 MBits | 16384 MBits |
| `PSU__DDRC__BG_ADDR_COUNT` | 1 | 2 |
| `PSU__DDRC__ROW_ADDR_COUNT` | 16 | 17 |
| `PSU__DDRC__CL`, `T_RCD`, `T_RP` | 16 | 17 |
| `PSU__DDRC__BRC_MAPPING` | ROW_BANK_COL | BANK_ROW_COL |
| `PSU__UART0__PERIPHERAL__IO` | MIO 34..35 | (UART1 32..33) |
| `PSU__SD0__SLOT_TYPE` | eMMC | SD 2.0 |

The FSBL configured the DDR controller for 64-bit / 8 Gb / 2400R with
the wrong timings. The training silently failed, leaving the PS unable
to fetch instructions from DRAM. The UART was also mis-routed
(UART0 vs UART1 on a different MIO pair), so even if any output had
been produced, it would not have reached the FTDI.

### Lesson

Tutorials that reuse the same SoC across boards are not portable
without rewriting the entire PSU init block. Same silicon, different
PCB, completely different bring-up.

---

## Stage 3 — Pivot to PYNQ 3.1.1 + DPU-PYNQ

### Rationale at the time

If the PSU init is the hard part, why not use the official PYNQ image
from Real Digital? The PSU init is guaranteed correct, Linux is
already running, and DPU-PYNQ provides a Python interface to a Xilinx
DPU loaded as an overlay.

### Execution

- Downloaded `AUP-ZU3-3.1.1-8gb.zip` (the official image).
- Flashed to SD, booted, verified Jupyter, SSH, base overlay loading.
  Tagged as `v0.1-pynq-vanilla` in the sibling repo.
- Installed `pip3 install pynq-dpu` from PyPI (version 2.5.1).
- Manually downloaded `dpu.bit / dpu.hwh / dpu.xclbin` for **Ultra96v2**
  (same ZU3EG silicon, same B1600 DPU configuration) because AUP-ZU3
  is not in the supported boards list.
- `Overlay("dpu.bit")` loaded successfully. `DPUCZDX8G_1` visible in
  `ip_dict`.
- `overlay.load_model("dpu_mnist_classifier.xmodel")` **silently killed
  the Python kernel**.

### Diagnosis

Dug through:

- `show_dpu` failing with
  `[XRT] ERROR: can't read kds_custat sysfs node`.
- `xclbinutil --version` reporting XRT 2.17 on the system.
- `strings /usr/lib/libvart-runner.so*` showing
  `Xilinx vart-runner Version: 2.5.0` (compiled 2022).
- AMD documentation: XCL APIs (including `xclClose`) were deprecated in
  XRT 2.14, last supported in 2.15. The XRT 2.17 in PYNQ 3.1.1 no
  longer exports them publicly, breaking the VART 2.5 ABI.

### Patch attempt

Found the official AMD patch `vai3.5_kr260.zip` referenced in
`amd/Kria-RoboticsAI`. Applied it:

- Upgraded VART/XIR/unilog/target-factory/libvitis-ai-library from 2.5.0
  to 3.5.0.
- Copied `lack_lib` (mostly Boost 1.80 shims) to /usr/lib.
- Installed Boost 1.80 filesystem and system from Debian experimental
  via snapshot.debian.org (the version is not in any stable repo).
- Patched `LD_LIBRARY_PATH=/usr/lib` in `/etc/profile.d/pynq_venv.sh`.

The notebook now ran `DpuOverlay("dpu.bit")` cleanly and crashed
again at `load_model()` with:

```
[VART_RUNNER_CONSTRUCTION_FAIL][Cannot create runner]
cannot open library! lib=libvart-dpu-runner.so
error: /lib/libvart-xrt-device-handle.so.3: undefined symbol: xclClose
```

Same root cause, deeper layer.

### The official verdict

A search on the PYNQ forum surfaced this answer from
**joshgoldsmith** (PYNQ maintainer, Nov 2025):

> "DPU-PYNQ is not supported in PYNQ v3.1 due to vart compatibility.
> PYNQ v3.0.1 still supports it though, so I would suggest porting your
> board to that version instead."

That is, the runtime mismatch is acknowledged and there is no fix
planned. The supported path is PYNQ 3.0.1 with the 2022.1 toolchain
(Vivado, Vitis, PetaLinux). Estimated cost of porting: 10-15 days plus
risk, eating most of the thesis time budget.

### Lesson

The DPU is the most capable accelerator for this silicon (B1600 in
ZU3EG, ~440 GOPS peak, can run any quantised xmodel from DRAM), but it
ships behind a runtime stack (VART → XIR → XRT → ZOCL) that is tightly
versioned. Mixing PYNQ image version, VART version and XRT version is
brittle. When they go out of sync, "fixes" require either downgrading
everything or rebuilding multiple components from source.

The exploration is documented in detail in the sibling repository
`AUP-ZU3-DPU_PYNQ`, which is preserved as a record of the path.

---

## Stage 4 — hls4ml + ICTP reference (current path)

### Why hls4ml is structurally different

Both FINN and DPU represent two extremes of "how do I deploy a CNN on
an FPGA":

- **DPU**: a fixed, large IP that interprets compiled xmodels at
  runtime from DRAM. One bitstream, many models. Runtime-heavy.
- **FINN**: per-model dataflow accelerator with extreme quantisation
  (1-2 bits). One bitstream per model. Runtime-light. Limited numeric
  precision.

hls4ml sits between them:

- **Per-model**: each trained model produces its own HLS IP, embedded
  in its own bitstream. Like FINN.
- **Configurable precision**: `ap_fixed<N,I>` is a free parameter; can
  go from binary up to 24-bit fixed-point or beyond. Unlike FINN
  (which is 1-2 bits) and unlike DPU (which is INT8 fixed).
- **Runtime-light**: pure `pynq.Overlay()` + AXI-Stream + DMA. Unlike
  DPU.
- **Industrial / academic use**: widely used at CERN for ultra-low
  latency inference. Actively maintained (v1.1.0 current). Unlike
  BNN-PYNQ or QNN-MO-PYNQ (both archived).

### The ICTP reference

The [ICTP/AUP-ZU3-HLS4ML](https://github.com/ICTP/AUP-ZU3-HLS4ML)
repository, sponsored by the **AMD University Program** (the same
program that backs the AUP-ZU3 board), provides three reference
projects (MLP, CNN, GRU) with end-to-end notebooks specifically for
this board. This is the upstream we follow.

### v0.1 result

The ICTP MNIST CNN reference was deployed unmodified on the AUP-ZU3
with the PYNQ 3.1.1 image. Inference over the 9999-sample test set
ran in seconds with **90.54 % accuracy**, with no VART, no XRT
extensions, no patches.

Documented in `docs/01-v0.1-validation.md`. Tagged
`v0.1-mnist-reproduction`.

This is the first time in the entire TFM effort that an
FPGA-accelerated CNN produces a numerical result end-to-end on the
target hardware.

---

## Comparison table: the three approaches

| Aspect | FINN | DPU-PYNQ | hls4ml |
|---|---|---|---|
| Precision | 1-2 bit weights/activations | INT8 standard | configurable `ap_fixed<N,I>` |
| Bitstream | per-model | one, runs any xmodel | per-model |
| Runtime stack | none (pynq.Overlay) | VART + XIR + XRT + ZOCL + libvart-* | none (pynq.Overlay) |
| PYNQ 3.1.1 support | yes | **broken** | yes |
| Vivado version | 2024.1 OK | needs 2022.1 | 2024.1 OK |
| DSP usage | very low (XNOR) | high (multiplier array) | configurable via reuse_factor |
| Model size limit | small | huge (DRAM) | small (on-chip BRAM/URAM) |
| Latency | very low (µs) | low (~ms) | very low (µs to low ms) |
| MobileNet feasible | no | yes | yes if alpha small + low res |
| Academic / industrial use | research / Xilinx | mainstream Xilinx | CERN, AMD UP |
| Maintenance status | active | **archived Aug 2025** | actively maintained |
| Verified on AUP-ZU3 | yes (prior work) | no (failed) | **yes (v0.1)** |

## Why hls4ml is the chosen path now

1. It works on the existing PYNQ 3.1.1 image with zero runtime patches.
2. It uses the Vivado 2024.1 already installed.
3. The AMD University Program backs both the AUP-ZU3 board and the
   ICTP reference repository — a single coherent ecosystem.
4. Precision is configurable: we can run 16-bit fixed-point (close to
   DPU's INT8 in practice) or push toward smaller representations for
   larger models.
5. The architecture is per-model: it makes the engineering of the
   target model the central exercise of the thesis, which is exactly
   what a TFM should be.

## What this teaches for the thesis chapter on FPGA AI deployment

The exploration produced three concrete lessons worth keeping for the
thesis discussion:

1. **The runtime stack is often the gating factor.** The DPU is
   architecturally fine for ZU3EG. What broke deployment was a chain
   of ABI mismatches in VART/XRT/Boost, not the hardware.

2. **Tutorials don't port across PCBs with the same SoC.** The
   Hackster TCL was technically correct for its target board but
   structurally wrong for ours, in ways that produce silent failures
   (DDR training fails before the UART can speak).

3. **The right abstraction depends on what you control.** If you
   control the model: per-model flows (FINN, hls4ml) are fine. If you
   want to swap models at runtime: DPU is the only realistic option.
   The TFM controls the model, so per-model is the right choice.

---

## State as of 2026-06-09

- `v0.1-mnist-reproduction` tagged and pushed to GitHub.
- Plan for v0.2 to v0.4 + v1.0 documented in the project README and in
  task tracking.
- Deadline: 2026-07-02. 23 days remaining.
- Time budget allocation (planned):
  - v0.2 own CIFAR-10 CNN: 7 days
  - v0.3 deeper CIFAR-10 CNN: 5 days
  - v0.4 MobileNet-style stretch: 5 days (optional)
  - v1.0 thesis writing: 6 days
