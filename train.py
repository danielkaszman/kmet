import numpy as np
import os
import tensorflow as tf
import keras
from pathlib import Path

cwd = Path.cwd()
data_dir = Path(os.path.join(cwd, "Numbers"))

batch_size = 16
img_height = 140
img_width = 90

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels="inferred",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=batch_size).cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 10

model = tf.keras.Sequential([
  tf.keras.Input(shape=(140, 90, 1)),
  tf.keras.layers.RandomRotation(0.3),
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                # keras.metrics.FalseNegatives(),
              ])

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_ds, epochs=10, callbacks=[early_stopping])

model.save("model2.keras", save_format='keras')

model.get_metrics_result()
