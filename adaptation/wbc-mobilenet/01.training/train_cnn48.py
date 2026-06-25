#!/usr/bin/env python3
"""
v0.3 (48x48 path) — train a quantised standard CNN student on BloodMNIST
resized to 48x48, with knowledge distillation from a float teacher.

Why this and not the depthwise MobileNet: QKeras 0.9 fails to train
QDepthwiseConv2D (stuck at ~18%). A standard quantised CNN trains fine
(~76% in 8 epochs). At 48x48 with a narrow channel plan (16->24->48) and
no large dense head, the MAC count stays ~2x the CIFAR v0.2 design, which
should fit the device once Vivado maps it (HLS over-estimates).

Saves models/distilled_cnn48.h5 for the hls4ml stage.

Run (host, neuralEnv10):
  cd 01.training && python train_cnn48.py
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
SIZE = 48

qb   = quantized_bits(bits=8, integer=3, alpha=1)
qrel = quantized_relu(bits=8, integer=3)


def load_data():
    tr = np.load(os.path.join(DATA, "bloodmnist64_train.npz"))
    te = np.load(os.path.join(DATA, "bloodmnist64_test.npz"))
    x_train = tf.image.resize(tr["x"], (SIZE, SIZE)).numpy()
    x_test  = tf.image.resize(te["x"], (SIZE, SIZE)).numpy()
    y_train = keras.utils.to_categorical(tr["y"], N_CLASSES)
    y_test  = keras.utils.to_categorical(te["y"], N_CLASSES)
    print(f"train {x_train.shape}, test {x_test.shape}")
    return x_train, y_train, x_test, y_test, te["y"]


def build_teacher():
    return models.Sequential([
        layers.Input((SIZE, SIZE, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 48 -> 24
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 24 -> 12
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 12 -> 6
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])


def build_student():
    # Narrow standard quantised CNN. Channels 16->24->48, three stride-2
    # convolutions, then global average pool + tiny dense head.
    inp = layers.Input(shape=(SIZE, SIZE, 3), name='conv1_input')
    x = QConv2D(16, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv1')(inp)
    x = QActivation(qrel, name='act1')(x)           # 48 -> 24
    x = QConv2D(24, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv2')(x)
    x = QActivation(qrel, name='act2')(x)           # 24 -> 12
    x = QConv2D(48, 3, strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='conv3')(x)
    x = QActivation(qrel, name='act3')(x)           # 12 -> 6
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

    teacher = build_teacher()
    teacher.compile(optimizer=optimizers.Adam(1e-3),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n=== Training teacher ===")
    teacher.fit(x_train, y_train, batch_size=BATCH, epochs=20,
                validation_split=0.1, verbose=2,
                callbacks=[callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=4,
                    restore_best_weights=True)])
    acc_t = teacher.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Teacher test accuracy: {acc_t*100:.2f}%")

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
    distiller.fit(x_train, y_train, batch_size=BATCH, epochs=35,
                  validation_split=0.1, verbose=2)
    print(f"Student training time: {time.time()-t0:.0f}s")

    y_pred = np.argmax(student.predict(x_test, verbose=0), axis=1)
    acc_s = accuracy_score(y_test_int, y_pred)
    print(f"\nStudent test accuracy (QKeras, software): {acc_s*100:.2f}%")
    print(f"Teacher test accuracy (float):            {acc_t*100:.2f}%")

    out_path = os.path.join(HERE, "models", "distilled_cnn48.h5")
    student.save(out_path)
    print(f"Saved student to {out_path}")


if __name__ == "__main__":
    main()
