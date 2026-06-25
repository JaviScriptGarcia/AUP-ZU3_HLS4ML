#!/usr/bin/env python3
"""
Download and prepare BloodMNIST (64x64 RGB) from MedMNIST v2.

BloodMNIST: 17,092 microscope images of normal blood cells, 8 classes
(neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature
granulocyte, erythroblast, platelet). License: CC BY 4.0.

This is the v0.3 dataset: a real medical patch-classification problem at
64x64 RGB, ~4x the pixels of the CIFAR-10 experiment, chosen to actually
load the FPGA and to justify the depthwise-separable (MobileNet) design.

Output (saved into this folder):
  - bloodmnist64_train.npz   (x_train, y_train)
  - bloodmnist64_test.npz    (x_test,  y_test)
  - bloodmnist64_test.csv    (flattened, for on-board inference like v0.2)

Run (host, neuralEnv10 env):
  python prepare_bloodmnist.py
"""
import os
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 64  # MedMNIST+ resolution


def main():
    try:
        from medmnist import BloodMNIST
        from medmnist import INFO
    except ImportError:
        raise SystemExit(
            "medmnist not installed. Run:  pip install medmnist\n"
            "(small pure-python package, CC BY 4.0 data / Apache-2.0 code)"
        )

    info = INFO["bloodmnist"]
    n_classes = len(info["label"])
    print(f"BloodMNIST: {n_classes} classes")
    for k, v in info["label"].items():
        print(f"  {k}: {v}")

    # download=True fetches the .npz the first time; size=64 -> MedMNIST+
    splits = {}
    for split in ("train", "val", "test"):
        ds = BloodMNIST(split=split, download=True, size=SIZE, root=OUT_DIR)
        x = ds.imgs.astype(np.float32) / 255.0          # (N, 64, 64, 3)
        y = ds.labels.flatten().astype(np.int64)        # (N,)
        splits[split] = (x, y)
        print(f"  {split}: x={x.shape}, y={y.shape}, "
              f"min/max={x.min():.2f}/{x.max():.2f}")

    # Merge train+val for training (we keep a validation split inside the
    # training notebook), keep test separate.
    x_train = np.concatenate([splits["train"][0], splits["val"][0]], axis=0)
    y_train = np.concatenate([splits["train"][1], splits["val"][1]], axis=0)
    x_test, y_test = splits["test"]

    np.savez_compressed(os.path.join(OUT_DIR, "bloodmnist64_train.npz"),
                        x=x_train, y=y_train)
    np.savez_compressed(os.path.join(OUT_DIR, "bloodmnist64_test.npz"),
                        x=x_test, y=y_test)
    print(f"Saved train ({len(x_train)}) and test ({len(x_test)}) npz.")

    # CSV for on-board inference: 64*64*3 = 12288 float columns + label.
    # Channels run fastest (HWC, C innermost), matching the AXI packing.
    flat = x_test.reshape(len(x_test), -1)
    csv = np.hstack([flat, y_test[:, None].astype(np.float32)])
    csv_path = os.path.join(OUT_DIR, "bloodmnist64_test.csv")
    # Save without pandas to avoid a heavy dependency; header optional.
    np.savetxt(csv_path, csv, delimiter=",", fmt="%.6f")
    print(f"Saved test CSV: {csv_path} "
          f"({csv.shape[0]} rows x {csv.shape[1]} cols)")


if __name__ == "__main__":
    main()
