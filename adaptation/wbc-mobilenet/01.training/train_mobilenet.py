#!/usr/bin/env python3
"""
Train a MobileNet-mini (depthwise separable) student on BloodMNIST 64x64
with knowledge distillation from a larger teacher, quantisation-aware
(QKeras). Saves the distilled student to models/distilled_wbc.h5 for the
hls4ml stage.

This is the v0.3 experiment: a real medical patch-classification task at
64x64 RGB, chosen to actually load the FPGA. The depthwise-separable
design is what keeps the MACs within the device budget at this input size.

Run (host, neuralEnv10 env):
  cd 01.training && python train_mobilenet.py

Inputs:  ../00.dataset/bloodmnist64_train.npz, bloodmnist64_test.npz
Output:  models/distilled_wbc.h5
"""
import os
import time
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from qkeras import (QConv2D, QDepthwiseConv2D, QActivation,
                    quantized_bits, quantized_relu)
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "00.dataset")
N_CLASSES = 8
BATCH = 128

# -- Quantisers: 8 bits, 3 integer bits (range ~ [-4, 4)) --
qb   = quantized_bits(bits=8, integer=3, alpha=1)
qrel = quantized_relu(bits=8, integer=3)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data():
    tr = np.load(os.path.join(DATA, "bloodmnist64_train.npz"))
    te = np.load(os.path.join(DATA, "bloodmnist64_test.npz"))
    x_train, y_train_int = tr["x"], tr["y"]
    x_test,  y_test_int  = te["x"], te["y"]
    y_train = keras.utils.to_categorical(y_train_int, N_CLASSES)
    y_test  = keras.utils.to_categorical(y_test_int,  N_CLASSES)
    print(f"train {x_train.shape}, test {x_test.shape}")
    return x_train, y_train, x_test, y_test, y_test_int


# ----------------------------------------------------------------------
# Teacher (float, standard convolutions, larger)
# ----------------------------------------------------------------------
def build_teacher():
    return models.Sequential([
        layers.Input((64, 64, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 64 -> 32
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 32 -> 16
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),                      # 16 -> 8
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])


# ----------------------------------------------------------------------
# Student: MobileNet-mini (quantised depthwise separable)
# ----------------------------------------------------------------------
def dw_sep_block(x, cout, stride, name):
    x = QDepthwiseConv2D((3, 3), strides=stride, padding='same',
                         depthwise_quantizer=qb, name=f'{name}_dw')(x)
    x = QActivation(qrel, name=f'{name}_dwact')(x)
    x = QConv2D(cout, (1, 1), padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name=f'{name}_pw')(x)
    x = QActivation(qrel, name=f'{name}_pwact')(x)
    return x


def build_student():
    inp = layers.Input(shape=(64, 64, 3), name='conv1_input')
    x = QConv2D(16, (3, 3), strides=2, padding='same',
                kernel_quantizer=qb, bias_quantizer=qb, name='stem')(inp)
    x = QActivation(qrel, name='stem_act')(x)      # 64 -> 32
    x = dw_sep_block(x, 32,  2, 'b1')              # 32 -> 16
    x = dw_sep_block(x, 64,  2, 'b2')              # 16 -> 8
    x = dw_sep_block(x, 128, 2, 'b3')              # 8  -> 4
    x = dw_sep_block(x, 128, 1, 'b4')              # 4  -> 4
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(N_CLASSES, name='output')(x)
    out = layers.Activation('softmax', name='softmax')(x)
    return models.Model(inp, out)


# ----------------------------------------------------------------------
# Knowledge distillation
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def main():
    x_train, y_train, x_test, y_test, y_test_int = load_data()
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

    # --- Teacher ---
    teacher = build_teacher()
    teacher.compile(optimizer=optimizers.Adam(1e-3),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n=== Training teacher ===")
    t0 = time.time()
    teacher.fit(x_train, y_train, batch_size=BATCH, epochs=25,
                validation_split=0.1, verbose=2,
                callbacks=[callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=5,
                    restore_best_weights=True)])
    acc_t = teacher.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Teacher test accuracy: {acc_t*100:.2f}%  ({time.time()-t0:.0f}s)")
    teacher.save(os.path.join(HERE, "models", "teacher_wbc.h5"))

    # --- Student via distillation ---
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
    distiller.fit(x_train, y_train, batch_size=BATCH, epochs=40,
                  validation_split=0.1, verbose=2)
    print(f"Student training time: {time.time()-t0:.0f}s")

    y_pred = np.argmax(student.predict(x_test, verbose=0), axis=1)
    acc_s = accuracy_score(y_test_int, y_pred)
    print(f"\nStudent test accuracy (QKeras, software): {acc_s*100:.2f}%")
    print(f"Teacher test accuracy (float):            {acc_t*100:.2f}%")

    out_path = os.path.join(HERE, "models", "distilled_wbc.h5")
    student.save(out_path)
    print(f"Saved student to {out_path}")


if __name__ == "__main__":
    main()
