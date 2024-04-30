import numpy as np
import os
import tensorflow as tf
from pathlib import Path

cwd = Path.cwd()
data_dir = Path(os.path.join(cwd, "Numbers"))

batch_size = 16
img_height = 140
img_width = 90

train_ds = tf.keras.utils.image_dataset_from_directory(
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
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(140, 90, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_ds, epochs=3, callbacks=[early_stopping])

#model.save("model2.h5", save_format='h5')
