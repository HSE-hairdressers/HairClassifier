import logging
import tensorflow as tf
import cv2
import numpy as np

from src.utils.constants import MODEL_NAME, CLASSES_FILE_NAME
from src.segmentation.imagecutter import ImageCutter


class RestoredModel:
    """
    Class for using a pretrained model, which was saved in the memory
    """
    def __init__(self, model_path):
        """
        Initialization function
        :param model_path: absolute path of the folder with pretrained model
        """
        self.cutter = ImageCutter()
        self.model = tf.keras.models.load_model(model_path + MODEL_NAME)
        # Retrieve class names
        with (open(model_path + MODEL_NAME + "/" + CLASSES_FILE_NAME)) as infile:
            self.class_names = [name.rstrip('\n') for name in infile.readlines()]

    def predict(self, image) -> str:
        """
        Prediction function
        :param image: input image
        :return: the most probable class to which image belongs
        """
        # Preprocess image
        image = self.cutter.get_face(image)
        image = self.cutter.cut_face_out(image)
        # Resize image to match with input model
        image = cv2.resize(image, (180, 180))
        # Convert th image to Tensor
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, 0)
        logging.debug("Classification started")
        predictions = self.model.predict(image_tensor, use_multiprocessing=True)
        # Choose class with the highest probability
        result = self.class_names[np.argmax(predictions[0])]
        # Print logs for debug
        logging.info("Prediction results:")
        for i in range(len(predictions[0])):
            logging.info(self.class_names[i] + " " + str(predictions[0][i]))
        print()
        return result
