import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib

from src.imagecutter import ImageCutter
from src.constants import MODEL_NAME, CLASSES_FILE_NAME


class ImageClassifier:
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS = 1
    DATA_AUGMENTATION = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(IMG_HEIGHT,
                                           IMG_WIDTH,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    def __init__(self, data_set_path):
        self.data_dir = pathlib.Path(data_set_path)
        self.cutter = ImageCutter()
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
        train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=self.AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        normalization_layer = layers.Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        num_classes = len(self.class_names)

        self.model = Sequential([
            self.DATA_AUGMENTATION,
            layers.Dropout(0.1),
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=self.EPOCHS)

    def save(self, path):
        tf.keras.models.save_model(self.model, path + MODEL_NAME)
        with (open(path + CLASSES_FILE_NAME, "wt")) as outfile:
            outfile.writelines([name + "\n" for name in self.class_names])
