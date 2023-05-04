from functools import partial

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.models import Sequential
from albumentations import (
    Compose, RandomBrightnessContrast, ImageCompression, HueSaturationValue, HorizontalFlip,
    Rotate
)
import pathlib

from src.constants import MODEL_NAME, CLASSES_FILE_NAME


class ImageClassifier:
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS = 5
    DATA_AUGMENTATION = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def __init__(self, dataset_path):
        self.data_dir = pathlib.Path(dataset_path)
        self.model = None
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=13,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=13,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)

        self.class_names = self.train_ds.class_names

    def fit(self):
        train_ds = self.train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        # train_ds = train_ds.map(partial(self.process_data, img_size=180),
        #                         num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
        # train_ds = train_ds.map(self.set_shapes, num_parallel_calls=self.AUTOTUNE).batch(32).prefetch(self.AUTOTUNE)

        val_ds = self.val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        num_classes = len(self.class_names)
        base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                    include_top=False,
                                                    input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(180, 180, 3))
        x = self.DATA_AUGMENTATION(inputs)
        # x = layers.Dropout(0.2)(x)
        x = base_model(x, training=False)
        x = layers.Dropout(0.5)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1000, activation='relu')(x)
        predictions = layers.Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=predictions)
        # self.model = Sequential([
        #     self.DATA_AUGMENTATION,
        #     layers.Dropout(0.2),
        #     layers.Rescaling(1. / 255),
        #     layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(64, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Dropout(0.4),
        #     layers.Flatten(),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(num_classes, activation='softmax')
        # ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['accuracy'])
        self.model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=self.EPOCHS)

    def save(self, path):
        tf.keras.models.save_model(self.model, path + MODEL_NAME)
        with (open(path + MODEL_NAME + "/" + CLASSES_FILE_NAME, "wt")) as outfile:
            outfile.writelines([name + "\n" for name in self.class_names])
