import mediapipe as mp
import cv2
import numpy as np


class Cutter:
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.__face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.__selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def __get_face_zone(self, image):
        height, width = image.shape[:2]
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = self.__face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            bbox_points = {
                "xmin": max(int(bbox.xmin * width - bbox.width * width * 0.4), 0),
                "ymin": max(int(bbox.ymin * height - bbox.height * height * 0.7), 0),
                "xmax": min(int(bbox.width * width * 1.4 + bbox.xmin * width), width - 1),
                "ymax": min(int(bbox.height * height * 1.3 + bbox.ymin * height), height - 1)
            }
        return image[bbox_points["ymin"]:bbox_points["ymax"], bbox_points["xmin"]:bbox_points["xmax"]]

    def __set_black_background(self, image):
        height, width, channel = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get the result
        results = self.__selfie_segmentation.process(image_rgb)
        # extract segmented mask
        mask = results.segmentation_mask
        # it returns true or false where the condition applies in the mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        # resize the background image to the same size of the original frame
        black = np.zeros((height, width, 3), dtype="uint8")
        # combine frame and background image using the condition
        return np.where(condition, image, black)

    def get_face(self, image):
        new_size = (500, int(image.shape[0] * 500 / image.shape[1]))
        image_resized = cv2.resize(image, new_size)

        image_cut = self.__get_face_zone(image_resized)

        new_size = (180, int(image_cut.shape[0] * 180 / image_cut.shape[1]))
        image_cut_resized = cv2.resize(image_cut, new_size)

        face = self.__set_black_background(image_cut_resized)

        return face
