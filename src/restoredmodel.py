import tensorflow as tf
import cv2
import numpy as np

from src.constants import MODEL_NAME, CLASSES_FILE_NAME
from src.imagecutter import ImageCutter


class RestoredModel:
    def __init__(self, model_path):
        self.cutter = ImageCutter()
        self.model = tf.keras.models.load_model(model_path + MODEL_NAME)
        with (open(model_path + CLASSES_FILE_NAME)) as infile:
            self.class_names = [name.rstrip('\n') for name in infile.readlines()]

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
