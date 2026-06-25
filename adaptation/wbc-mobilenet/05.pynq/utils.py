"""
Utility helpers for the BloodMNIST (white blood cell) PYNQ inference
notebook, MobileNet-mini accelerator.

Same fixed-point scheme as the v0.2 CIFAR-10 design: `ap_fixed<16, 6>`
(16 total bits, 6 integer bits, 10 fractional bits), so FX_FRAC_BITS = 10.

Difference vs CIFAR-10: each image is 64x64x3, flattened to 12288 values
in row-major (H, W, C) order with C running fastest, matching the Keras
channels-last layout used during training.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixed-point conversion (fixed<24,12>: 12 integer bits, 12 fractional bits)
# This network needs more dynamic range than the CIFAR design: a precision
# sweep showed fixed<16,6> collapsed it to ~10% while fixed<24,12> keeps 83%.
# ---------------------------------------------------------------------------
FX_TOTAL_BITS = 24
FX_INT_BITS   = 12
FX_FRAC_BITS  = FX_TOTAL_BITS - FX_INT_BITS  # = 12
FX_MASK       = (1 << FX_TOTAL_BITS) - 1     # 0xFFFFFF
FX_SIGN       = 1 << (FX_TOTAL_BITS - 1)     # 0x800000

WBC_PIXELS = 48 * 48 * 3  # 6912


def float_to_axis_word(f):
    fx = int(round(f * (1 << FX_FRAC_BITS))) & FX_MASK
    return np.uint32(fx)


def axis_word_to_float(word):
    raw = int(word) & FX_MASK
    if raw & FX_SIGN:
        raw -= (1 << FX_TOTAL_BITS)
    return raw / (1 << FX_FRAC_BITS)


# ---------------------------------------------------------------------------
# Image packing
# ---------------------------------------------------------------------------

def image_to_axis_buffer(img, buffer):
    """
    Fill an already-allocated DMA buffer with one 48x48x3 image.

    Vectorised over the whole image (6912 values) for speed.

    Parameters
    ----------
    img : np.ndarray of shape (48, 48, 3) or (6912,)
        Channels-last, normalised to [0, 1].
    buffer : pynq.allocate buffer of shape (6912,) dtype uint32
    """
    flat = np.asarray(img, dtype=np.float32).reshape(-1)
    if flat.shape[0] != WBC_PIXELS:
        raise ValueError(
            f"Expected {WBC_PIXELS} values per image, got {flat.shape[0]}"
        )
    # float -> fixed<24,12>: scale, round, mask to 24 bits
    scaled = np.round(flat * (1 << FX_FRAC_BITS)).astype(np.int64)
    buffer[:] = (scaled & FX_MASK).astype(np.uint32)


def axis_buffer_to_floats(buffer):
    """Vectorised inverse: low 24 bits of uint32 words -> float32 scores."""
    raw = np.asarray(buffer, dtype=np.int64) & FX_MASK
    signed = np.where(raw >= FX_SIGN, raw - (1 << FX_TOTAL_BITS), raw)
    return signed.astype(np.float32) / (1 << FX_FRAC_BITS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, class_names=None, normalize=False, cmap="Blues"):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    if normalize:
        cm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            txt = f"{value:.2f}" if normalize else str(value)
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if value > thresh else "black")
    plt.tight_layout()
    plt.show()


# BloodMNIST class labels (MedMNIST v2 order)
WBC_CLASSES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]
