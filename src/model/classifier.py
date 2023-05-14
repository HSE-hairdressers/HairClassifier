import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.models import Sequential
import pathlib
from src.utils.constants import MODEL_NAME, CLASSES_FILE_NAME


class ImageClassifier:
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS = 10
    DATA_AUGMENTATION = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def __init__(self):
        """
        Initialization function
        """
        self.model = None
        self.class_names = None

    def fit(self, dataset_path, pretrained=False) -> "ImageClassifier":
        """
        Classifier training function
        :param pretrained: flag for transfer learning
        :param dataset_path: path to dataset folder
        :return: trained model
        """
        data_dir = pathlib.Path(dataset_path)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=13,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=13,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)
        self.class_names = train_ds.class_names
        train_ds = train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        num_classes = len(self.class_names)

        if pretrained:
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
        else:
            self.model = Sequential([
                self.DATA_AUGMENTATION,
                layers.Dropout(0.2),
                layers.Rescaling(1. / 255),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.4),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           metrics=['accuracy'])
        self.model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=self.EPOCHS)
        return self

    def save(self, path):
        """
        Method to save trained model
        :param path: path, where folder with model should be saved
        """
        if self.model is None:
            return
        tf.keras.models.save_model(self.model, path + MODEL_NAME)
        with (open(path + MODEL_NAME + "/" + CLASSES_FILE_NAME, "wt")) as outfile:
            outfile.writelines([name + "\n" for name in self.class_names])
