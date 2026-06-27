#!/usr/bin/env python3
"""
v0.4 (64x64 path) — train a quantised standard CNN student on BloodMNIST
at native 64x64, with knowledge distillation from a float teacher.

Improvements over the v0.3 48x48 training (train_cnn48.py):
  * Native 64x64 (no resize): BRAM is only ~41% used at 48x48, so the
    larger feature maps fit comfortably; the binding resource is DSP,
    which resolution barely touches.
  * Wider student (channels 24->32->64 vs 16->24->48): more capacity for
    accuracy. The extra DSP demand is meant to be absorbed by a higher
    reuse factor at the hls4ml stage (see convert_cnn64.py).
  * Early stopping on the STUDENT with restore_best_weights (v0.3 only had
    it on the teacher; the student ran a fixed 35 epochs).
  * Data augmentation (flips, small rotation/zoom) — cheap accuracy on
    microscopy images.
  * LR scheduling (ReduceLROnPlateau).

Still standard (not depthwise) conv: QKeras 0.9 cannot train
QDepthwiseConv2D (stuck ~18%).

Saves models/distilled_cnn64.h5 for the hls4ml stage.

Run (host, neuralEnv10):
  cd 01.training && PYTHONNOUSERSITE=1 python train_cnn64.py
"""
import os
import time
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
    """Light augmentation appropriate for cell microscopy (orientation
    is not semantically meaningful, so flips/rotations are safe)."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    # small brightness/contrast jitter
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
    return models.Sequential([
        layers.Input((SIZE, SIZE, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 64 -> 32
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 32 -> 16
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 16 -> 8
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])


def build_student():
    # Wider than v0.3: channels 24->32->64, three stride-2 convolutions,
    # global average pool + tiny dense head. At 64x64 the three stride-2
    # convs take 64->32->16->8.
    inp = layers.Input(shape=(SIZE, SIZE, 3), name='conv1_input')
    x = QConv2D(24, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv1')(inp)
    x = QActivation(qrel, name='act1')(x)           # 64 -> 32
    x = QConv2D(32, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv2')(x)
    x = QActivation(qrel, name='act2')(x)           # 32 -> 16
    x = QConv2D(64, 3, strides=2, padding='same',
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


def main():
    x_train, y_train, x_test, y_test, y_test_int = load_data()
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

    # Hold out 10% of train for validation (manual split so both teacher
    # and student see the same split, and so tf.data augmentation only
    # touches the training partition).
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
    print("\n=== Training teacher ===")
    teacher.fit(train_ds, validation_data=val_ds, epochs=25, verbose=2,
                callbacks=[
                    callbacks.EarlyStopping(monitor='val_accuracy',
                        patience=5, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                        factor=0.5, patience=3, min_lr=1e-5, verbose=1),
                ])
    acc_t = teacher.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Teacher test accuracy: {acc_t*100:.2f}%")

    # ---- Student (KD) ----
    student = build_student()
    student.summary()
    print(f"Student params: {student.count_params():,}")

    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=optimizers.Adam(1e-3),
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        distill_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.3, temperature=4)
    print("\n=== Training student (KD) ===")
    t0 = time.time()
    distiller.fit(train_ds, validation_data=val_ds, epochs=60, verbose=2,
                  callbacks=[
                      callbacks.EarlyStopping(monitor='val_accuracy',
                          patience=8, restore_best_weights=True),
                      callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                          factor=0.5, patience=4, min_lr=1e-5, verbose=1),
                  ])
    print(f"Student training time: {time.time()-t0:.0f}s")

    y_pred = np.argmax(student.predict(x_test, verbose=0), axis=1)
    acc_s = accuracy_score(y_test_int, y_pred)
    print(f"\nStudent test accuracy (QKeras, software): {acc_s*100:.2f}%")
    print(f"Teacher test accuracy (float):            {acc_t*100:.2f}%")
    print(f"(v0.3 reference at 48x48 was 83.05%)")

    out_path = os.path.join(HERE, "models", "distilled_cnn64.h5")
    student.save(out_path)
    print(f"Saved student to {out_path}")


if __name__ == "__main__":
    main()
