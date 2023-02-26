import tensorflow as tf
import cv2
import pathlib
import numpy as np

from src.cutter import Cutter


class HairClassifier:
    def __init__(self, restored_model, data_set_path):
        self.data_dir = pathlib.Path(data_set_path)
        self.cutter = Cutter()
        self.model = restored_model
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir)
        self.class_names = train_ds.class_names

    def predict(self, image) -> str:
        image = self.cutter.get_face(image)
        # Resize image to match with input model
        image = cv2.resize(image, (180, 180))

        # Convert to Tensor of type float32 for example
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, 0)
        predictions = self.model.predict(
            image_tensor, use_multiprocessing=True)
        result = self.class_names[np.argmax(predictions[0])]
        return result
