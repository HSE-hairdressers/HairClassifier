import logging
import mediapipe as mp
import cv2
import numpy as np

MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_FACE_MESH = mp.solutions.face_mesh

WHITE = (255, 255, 255)


class ImageCutter:
    """
    Class for image preprocessing.
    This class cuts image to leave only the face zone, sets black background and cuts out face to leave only hairline.
    """
    WIDTH_DELTA = 0.4
    FOREHEAD_DELTA = 0.7
    CHIN_DELTA = 0.3
    MIN_CONFIDENCE = 0.5
    SIZE = 500
    SIZE_CUT = 180

    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.__face_detection = mp_face_detection.FaceDetection(model_selection=1,
                                                                min_detection_confidence=self.MIN_CONFIDENCE)
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.__selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.__drawing_spec = MP_DRAWING.DrawingSpec(thickness=1, circle_radius=1)

    def get_face_zone(self, image):
        """
        Method, which cuts image to leave only the zone with a face detection
        :param image: input image
        :return: cut image of the face
        """
        height, width = image.shape[:2]
        # Convert the BGR image to RGB
        results = self.__face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Check if any faces were detected
        if not results.detections:
            logging.debug('No face found')
            raise ValueError('No face')

        bbox_points = { "xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
        # Get Bounding Box coordinates of
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            bbox_points = {
                "xmin": max(int(bbox.xmin * width - bbox.width * width * self.WIDTH_DELTA), 0),
                "ymin": max(int(bbox.ymin * height - bbox.height * height * self.FOREHEAD_DELTA), 0),
                "xmax": min(int(bbox.width * width * (1 + self.WIDTH_DELTA) + bbox.xmin * width), width - 1),
                "ymax": min(int(bbox.height * height * (1 + self.CHIN_DELTA) + bbox.ymin * height), height - 1)
            }
        logging.debug('Face found. Image cut')
        return image[bbox_points["ymin"]:bbox_points["ymax"], bbox_points["xmin"]:bbox_points["xmax"]]

    def set_black_background(self, image):
        """
        Method, which sets black background using selfie segmentation
        :param image: input image
        :return: image with black background
        """
        height, width, channel = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logging.debug('Selfie segmentation.')
        # Get result of selfie segmentation
        results = self.__selfie_segmentation.process(image_rgb)
        # Extract segmented mask
        mask = results.segmentation_mask
        # The  mask returns true or false where the condition applies in the mask
        condition = np.stack((mask,) * 3, axis=-1) > self.MIN_CONFIDENCE
        # Create black background image of the same size as the original frame
        black = np.zeros((height, width, 3), dtype="uint8")
        logging.debug('Background is set to black.')
        # Combine frame and black background image using the condition
        return np.where(condition, image, black)

    def get_face(self, image):
        """
        Method, which resizes image, cuts it and sets background to black to leave only the face in the picture
        :param image: input image
        :return: face image with black background
        """
        # Resize the input image to reduce the time of cutting
        new_size = (self.SIZE, int(image.shape[0] * self.SIZE / image.shape[1]))
        image_resized = cv2.resize(image, new_size)

        logging.debug('Cutting face zone')
        image_cut = self.get_face_zone(image_resized)

        # Resize the image again to be of the same size as the images in the dataset
        new_size = (self.SIZE_CUT, int(image_cut.shape[0] * self.SIZE_CUT / image_cut.shape[1]))
        image_cut_resized = cv2.resize(image_cut, new_size)

        logging.debug('Setting black background')
        face = self.set_black_background(image_cut_resized)
        return face

    def cut_face_out(self, image):
        """
        Method, which cuts out the face by contour using face mesh
        :param image: input image
        :return: image without face
        """
        print('Cutting out face...')
        with MP_FACE_MESH.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.MIN_CONFIDENCE) as face_mesh:

            # Convert the BGR image to RGB and process it with face landmark function
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Create a black image of the size of the input image to draw face landmarks on it
            black = np.zeros(image.shape, dtype="uint8")
            new_mask = black.copy()
            # Draw face mesh landmarks with thick lines on the image.
            if not results.multi_face_landmarks:
                return
            for face_landmarks in results.multi_face_landmarks:
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_TESSELATION,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=13,
                        circle_radius=0
                    ))
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_CONTOURS,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=1,
                        circle_radius=0
                    ))
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_IRISES,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=WHITE,
                        thickness=11,
                        circle_radius=0
                    ))
            # Use black image with painted up face zone as a mask
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[new_mask == 0] = 1
            # Apply mask to image to cut out face
            hair = image * mask[:, :, np.newaxis]
        return hair
