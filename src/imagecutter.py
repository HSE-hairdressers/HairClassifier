import mediapipe as mp
import cv2
import numpy as np

MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_FACE_MESH = mp.solutions.face_mesh

class ImageCutter:
    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.__face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.__selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.__drawing_spec = MP_DRAWING.DrawingSpec(thickness=1, circle_radius=1)

    def __get_face_zone(self, image):
        height, width = image.shape[:2]
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = self.__face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print('No face found')
            raise ValueError('No face')
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            bbox_points = {
                "xmin": max(int(bbox.xmin * width - bbox.width * width * 0.4), 0),
                "ymin": max(int(bbox.ymin * height - bbox.height * height * 0.7), 0),
                "xmax": min(int(bbox.width * width * 1.4 + bbox.xmin * width), width - 1),
                "ymax": min(int(bbox.height * height * 1.3 + bbox.ymin * height), height - 1)
            }
        print('Face found. Image cut')
        return image[bbox_points["ymin"]:bbox_points["ymax"], bbox_points["xmin"]:bbox_points["xmax"]]

    def __set_black_background(self, image):
        height, width, channel = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get the result
        print('Selfie segmentation.')
        results = self.__selfie_segmentation.process(image_rgb)
        # extract segmented mask
        mask = results.segmentation_mask
        # it returns true or false where the condition applies in the mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        # resize the background image to the same size of the original frame
        black = np.zeros((height, width, 3), dtype="uint8")
        print('Background is set to black.')
        # combine frame and background image using the condition
        return np.where(condition, image, black)

    def get_face(self, image):
        print('Classification started')
        new_size = (500, int(image.shape[0] * 500 / image.shape[1]))
        image_resized = cv2.resize(image, new_size)

        print('Cutting face zone')
        image_cut = self.__get_face_zone(image_resized)

        new_size = (180, int(image_cut.shape[0] * 180 / image_cut.shape[1]))
        image_cut_resized = cv2.resize(image_cut, new_size)

        print('Setting black background')
        face = self.__set_black_background(image_cut_resized)

        return face

    def cut_face_out(self, image):
        print('Cutting out face...')
        with MP_FACE_MESH.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            black = np.zeros(image.shape, dtype="uint8")
            new_mask = black.copy()
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                return
            for face_landmarks in results.multi_face_landmarks:
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_TESSELATION,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=13,
                        circle_radius=0
                    ))
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_CONTOURS,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=0
                    ))
                MP_DRAWING.draw_landmarks(
                    image=new_mask,
                    landmark_list=face_landmarks,
                    connections=MP_FACE_MESH.FACEMESH_IRISES,
                    landmark_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=0
                    ),
                    connection_drawing_spec=MP_DRAWING.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=11,
                        circle_radius=0
                    ))

            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[new_mask == 0] = 1
            hair = image * mask[:, :, np.newaxis]
        return hair
