#!/usr/bin/env python3
"""
v0.5 (64x64 path) — train a LARGER quantised CNN student on BloodMNIST at
native 64x64, with knowledge distillation AND a hyperparameter search.

Why v0.5 (vs v0.4):
  v0.4 (channels 24->32->64) used only ~50% of the chip on hardware
  (Vivado measured: LUT 50.5%, DSP 43.6%, BRAM 57.4%). The brief for v0.5
  is "a network that gets closer to the FPGA resource limit while keeping
  throughput". We verified empirically that no other well-known MedMNIST
  dataset stresses the board harder with *real* detail (PathMNIST and
  DermaMNIST at 64x64 are interpolated from 28x28; DermaMNIST is also 67%
  one class). So we stay on BloodMNIST 64x64 and instead grow the network.

Sizing (resource-aware, from v0.4 measurements):
  BRAM at 57% is the binding resource -> ~1.7x headroom before it fills.
  We pick channels 32->48->96 (~1.96x the conv MACs of v0.4) which should
  push BRAM to ~75-85% and DSP to ~70% without hitting the routing wall.
  The wider channels add compute over the SAME real 64x64 pixels (no
  upsampling), which is the only honest way to stress the board here.

Hyperparameter search (explicit user request):
  A reproducible random search over learning rate, dropout/label-smoothing,
  and the distillation alpha/temperature. Each candidate trains a few
  epochs; the best (by val_accuracy) is then retrained to convergence with
  early stopping + restore_best_weights. Seeded for reproducibility; no
  extra dependency (no keras-tuner).

Layer names conv1/conv2/conv3/output are kept so the hls4ml reuse plan in
convert_cnn64_v05.py maps onto them unchanged.

Still standard (not depthwise) conv: QKeras 0.9 cannot train
QDepthwiseConv2D (stuck ~18%).

Saves models/distilled_cnn64_v05.h5 for the hls4ml stage.

Run (host, neuralEnv10):
  cd 01.training && PYTHONNOUSERSITE=1 python train_cnn64_v05.py
"""
import os
import time
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from qkeras import QConv2D, QActivation, quantized_bits, quantized_relu
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "00.dataset")
N_CLASSES = 8
BATCH = 128
SIZE = 64                       # native resolution, no resize

# Wider than v0.4 (24->32->64): channels 32->48->96, ~1.96x conv MACs.
CH = (32, 48, 96)

qb   = quantized_bits(bits=8, integer=3, alpha=1)
qrel = quantized_relu(bits=8, integer=3)


def load_data():
    tr = np.load(os.path.join(DATA, "bloodmnist64_train.npz"))
    te = np.load(os.path.join(DATA, "bloodmnist64_test.npz"))
    x_train = tr["x"].astype(np.float32)      # already 64x64x3, [0,1]
    x_test  = te["x"].astype(np.float32)
    y_train = keras.utils.to_categorical(tr["y"], N_CLASSES)
    y_test  = keras.utils.to_categorical(te["y"], N_CLASSES)
    print(f"train {x_train.shape}, test {x_test.shape}, "
          f"range [{x_train.min():.3f}, {x_train.max():.3f}]")
    return x_train, y_train, x_test, y_test, te["y"]


def augment(x, y):
    """Light augmentation appropriate for cell microscopy (orientation is
    not semantically meaningful, so flips/rotations are safe)."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, max_delta=0.08)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


def make_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(4096, seed=SEED).map(
            augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_teacher():
    # Wider teacher to match the wider student (better dark knowledge).
    return models.Sequential([
        layers.Input((SIZE, SIZE, 3)),
        layers.Conv2D(48, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 64 -> 32
        layers.Conv2D(96, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 32 -> 16
        layers.Conv2D(192, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 16 -> 8
        layers.GlobalAveragePooling2D(),
        layers.Dense(192, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])


def build_student():
    # Channels 32->48->96, three stride-2 convolutions (64->32->16->8),
    # global average pool + tiny dense head. Names match the hls4ml plan.
    c1, c2, c3 = CH
    inp = layers.Input(shape=(SIZE, SIZE, 3), name='conv1_input')
    x = QConv2D(c1, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv1')(inp)
    x = QActivation(qrel, name='act1')(x)           # 64 -> 32
    x = QConv2D(c2, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv2')(x)
    x = QActivation(qrel, name='act2')(x)           # 32 -> 16
    x = QConv2D(c3, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv3')(x)
    x = QActivation(qrel, name='act3')(x)           # 16 -> 8
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(N_CLASSES, name='output')(x)
    out = layers.Activation('softmax', name='softmax')(x)
    return models.Model(inp, out)


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, student_loss_fn, distill_loss_fn,
                alpha=0.3, temperature=4):
        super().compile(optimizer=optimizer, metrics=['accuracy'])
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_pred)
            t = self.temperature
            distill_loss = self.distill_loss_fn(
                tf.nn.softmax(tf.math.log(teacher_pred + 1e-9) / t, axis=1),
                tf.nn.softmax(tf.math.log(student_pred + 1e-9) / t, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_pred)
        return {**{m.name: m.result() for m in self.metrics},
                'student_loss': student_loss, 'distill_loss': distill_loss}

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_pred)
        self.compiled_metrics.update_state(y, student_pred)
        return {**{m.name: m.result() for m in self.metrics},
                'student_loss': student_loss}


def train_student(teacher, train_ds, val_ds, hp, epochs, verbose=0):
    """Build + KD-train one student under hyperparameters `hp`.
    Returns (student, best_val_acc)."""
    tf.keras.utils.set_random_seed(SEED)        # same init per candidate
    student = build_student()
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=optimizers.Adam(hp["lr"]),
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=hp["label_smoothing"]),
        distill_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=hp["alpha"], temperature=hp["temperature"])
    hist = distiller.fit(
        train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_accuracy',
                patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                factor=0.5, patience=4, min_lr=1e-5, verbose=0),
        ])
    best_val = float(np.max(hist.history['val_accuracy']))
    return student, best_val


# Hyperparameter search space. Random search keeps it cheap and unbiased.
SEARCH_SPACE = {
    "lr":              [3e-3, 1e-3, 5e-4],
    "alpha":           [0.2, 0.3, 0.5],      # weight on hard-label loss
    "temperature":     [3, 4, 6],            # distillation softening
    "label_smoothing": [0.0, 0.05, 0.1],
}
N_TRIALS = 6        # random configs sampled
SEARCH_EPOCHS = 18  # short probe per config
FINAL_EPOCHS = 70   # full retrain of the winner


def sample_configs(n, rng):
    keys = list(SEARCH_SPACE.keys())
    seen, out = set(), []
    while len(out) < n and len(seen) < np.prod([len(v) for v in SEARCH_SPACE.values()]):
        cfg = {k: SEARCH_SPACE[k][rng.integers(len(SEARCH_SPACE[k]))] for k in keys}
        key = tuple(cfg[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def main():
    x_train, y_train, x_test, y_test, y_test_int = load_data()
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

    # 10% holdout for validation (manual split: same split for teacher and
    # student, augmentation only on the training partition).
    n = x_train.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = int(0.1 * n)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    xtr, ytr = x_train[tr_i], y_train[tr_i]
    xva, yva = x_train[val_i], y_train[val_i]
    print(f"train split {xtr.shape[0]}, val split {xva.shape[0]}")

    train_ds = make_ds(xtr, ytr, training=True)
    val_ds   = make_ds(xva, yva, training=False)

    # ---- Teacher ----
    teacher = build_teacher()
    teacher.compile(optimizer=optimizers.Adam(1e-3),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n=== Training teacher (wider) ===")
    teacher.fit(train_ds, validation_data=val_ds, epochs=30, verbose=2,
                callbacks=[
                    callbacks.EarlyStopping(monitor='val_accuracy',
                        patience=6, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                        factor=0.5, patience=3, min_lr=1e-5, verbose=1),
                ])
    acc_t = teacher.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Teacher test accuracy: {acc_t*100:.2f}%")

    student0 = build_student()
    student0.summary()
    print(f"Student params: {student0.count_params():,}")

    # ---- Hyperparameter search (random) ----
    configs = sample_configs(N_TRIALS, rng)
    print(f"\n=== Hyperparameter search: {len(configs)} configs x "
          f"{SEARCH_EPOCHS} epochs ===")
    results = []
    for i, hp in enumerate(configs):
        t0 = time.time()
        _, val = train_student(teacher, train_ds, val_ds, hp,
                               epochs=SEARCH_EPOCHS, verbose=0)
        dt = time.time() - t0
        results.append((val, hp))
        print(f"  [{i+1}/{len(configs)}] val_acc={val*100:.2f}%  "
              f"({dt:.0f}s)  {hp}")

    results.sort(key=lambda r: r[0], reverse=True)
    best_val, best_hp = results[0]
    print(f"\nBest config (probe val_acc={best_val*100:.2f}%): {best_hp}")
    with open(os.path.join(HERE, "models", "v05_search.json"), "w") as f:
        json.dump([{"val_acc": v, "hp": hp} for v, hp in results], f, indent=2)

    # ---- Final retrain of the winner ----
    print(f"\n=== Final training of winner ({FINAL_EPOCHS} epochs) ===")
    t0 = time.time()
    student, _ = train_student(teacher, train_ds, val_ds, best_hp,
                               epochs=FINAL_EPOCHS, verbose=2)
    print(f"Final student training time: {time.time()-t0:.0f}s")

    y_pred = np.argmax(student.predict(x_test, verbose=0), axis=1)
    acc_s = accuracy_score(y_test_int, y_pred)
    print(f"\nStudent test accuracy (QKeras, software): {acc_s*100:.2f}%")
    print(f"Teacher test accuracy (float):            {acc_t*100:.2f}%")
    print(f"(v0.4 reference at 64x64 was 89.16% sw / 88.98% hardware)")

    out_path = os.path.join(HERE, "models", "distilled_cnn64_v05.h5")
    student.save(out_path)
    print(f"Saved student to {out_path}")


if __name__ == "__main__":
    main()
