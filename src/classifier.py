from functools import partial

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.models import Sequential
from albumentations import (
    Compose, RandomBrightnessContrast, ImageCompression, HueSaturationValue, HorizontalFlip,
    Rotate, RandomBrightness, RandomContrast
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
    TRANSFORMS = transforms = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),
            ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
    ])

    def augment(self, image, img_size):
        data = {"image": image}
        aug_data = self.TRANSFORMS(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img / 255.0, tf.float32)
        aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
        return aug_img

    def process_data(self, image, label, img_size):
        if image is None:
            return image, label
        aug_img = tf.numpy_function(func=self.augment, inp=[image, img_size], Tout=tf.float32)
        return aug_img, label

    def set_shapes(self, img, label, img_shape=(180, 180, 3)):
        img.set_shape(img_shape)
        return img, label

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

        inputs = tf.keras.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        x = self.DATA_AUGMENTATION(inputs)
        x = base_model(x, training=False)
        x = layers.Dropout(0.4)(x)
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
