"""
Utility helpers for the CIFAR-10 PYNQ inference notebook.

Key differences vs the MNIST utils (02-cnn/05.pynq/utils.py):

- The hls4ml model for CIFAR-10 uses `ap_fixed<16, 6>` precision
  (16 total bits, 6 integer bits, 10 fractional bits), whereas the
  ICTP MNIST reference used `ap_fixed<16, 12>` (4 fractional bits).
  Therefore `FX_FRAC_BITS` is 10 here, not 4.

- The dataset shape is (32, 32, 3) per image, flattened to 3072 values
  in row-major order (H, W, C) with C running fastest. This matches the
  Keras `image_data_format = 'channels_last'` convention used during
  training.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixed-point conversion
# ---------------------------------------------------------------------------

# The hls4ml model was generated with default_precision = 'fixed<16,6>'.
# - Total bits: 16
# - Integer bits (including sign): 6
# - Fractional bits: 10
FX_TOTAL_BITS = 16
FX_INT_BITS   = 6
FX_FRAC_BITS  = FX_TOTAL_BITS - FX_INT_BITS  # = 10


def float_to_axis_word(f):
    """
    Convert a Python float into the 16-bit fixed<16,6> representation,
    then pack into a 32-bit AXI word for DMA transfer (16 LSBs used,
    upper 16 bits zero).
    """
    fx = int(round(f * (1 << FX_FRAC_BITS))) & 0xFFFF
    return np.uint32(fx)


def axis_word_to_float(word):
    """
    Inverse of float_to_axis_word: interpret the low 16 bits of an AXI
    beat as a signed fixed<16,6> value and return its float representation.
    """
    raw = int(word) & 0xFFFF
    # Sign-extend the 16-bit value
    if raw & 0x8000:
        raw -= 0x10000
    return raw / (1 << FX_FRAC_BITS)


# ---------------------------------------------------------------------------
# Image packing
# ---------------------------------------------------------------------------

def image_to_axis_buffer(img, buffer):
    """
    Fill an already-allocated DMA buffer with a CIFAR-10 image.

    Vectorised over the whole image: 3072 individual float-to-fixed
    conversions in Python would be ~1-2 seconds per image; numpy does
    it in microseconds.

    Parameters
    ----------
    img : np.ndarray of shape (32, 32, 3) or (3072,)
        Input image. If 3D, it is expected to be in (H, W, C) order
        (channels last), matching the Keras model layout. Values
        should already be normalised to the range the model expects
        (typically [0, 1]).
    buffer : pynq.allocate buffer of shape (3072,) dtype uint32
        The buffer that will be sent through the DMA.
    """
    flat = np.asarray(img, dtype=np.float32).reshape(-1)
    if flat.shape[0] != 3072:
        raise ValueError(
            f"Expected 3072 values per image, got {flat.shape[0]}"
        )
    # float -> fixed<16,6>: multiply by 2^FRAC_BITS, round, mask to 16 bits
    scaled = np.round(flat * (1 << FX_FRAC_BITS)).astype(np.int32)
    buffer[:] = (scaled & 0xFFFF).astype(np.uint32)


def axis_buffer_to_floats(buffer):
    """
    Vectorised inverse of image_to_axis_buffer for an output buffer of
    fixed<16,6> values packed as the low 16 bits of uint32 words.

    Returns a numpy array of float32 scores.
    """
    raw = np.asarray(buffer, dtype=np.int32) & 0xFFFF
    # Sign-extend 16-bit values inside an int32
    signed = np.where(raw >= 0x8000, raw - 0x10000, raw)
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
    """
    Plot a confusion matrix with matplotlib.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    if normalize:
        cm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 6))
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
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=12,
                color="white" if value > thresh else "black",
            )

    plt.tight_layout()
    plt.show()


# Class labels for CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
