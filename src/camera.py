"""
Camera class definition.
"""

import cv2
import glob
import numpy as np


class Camera:
    """
    Responsible for removing distortions in images.
    """

    def __init__(self, camera_matrix, dist_coefficients):
        """
        Initializer.
        """
        self._camera_matrix = camera_matrix
        self._dist_coefficients = dist_coefficients

    @staticmethod
    def calibrate(calibration_images_dir, x_intersections, y_intersections):
        """
        Calibrates the camera basing on images stored in a given directory

        :param str calibration_images_dir: Directory containing chessboards required for calibration.
        :param int x_intersections       : Number of intersections in a single row.
        :param int y_intersections       : Number of intersections in a single column.

        :rtype: Camera
        """
        object_points = np.zeros((x_intersections * y_intersections, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:x_intersections, 0:y_intersections].T.reshape(-1, 2)

        real_object_points = []
        image_points = []

        for image_path in glob.glob(calibration_images_dir + "*.jpg"):
            image = cv2.imread(image_path)
            calibration_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            all_intersections_found, intersections = cv2.findChessboardCorners(
                calibration_image, (x_intersections, y_intersections)
            )

            if all_intersections_found:
                real_object_points.append(object_points)
                image_points.append(intersections)

        sample_image = cv2.imread(glob.glob(calibration_images_dir + "*.jpg")[0])
        image_size = (sample_image.shape[0], sample_image.shape[1])
        _, camera_matrix, dist_coefficients, _, _ = cv2.calibrateCamera(
            real_object_points, image_points, image_size, None, None
        )
        return Camera(camera_matrix, dist_coefficients)

    def correct_distortions(self, image):
        """
        Corrects distortion in a given image.

        :param np.ndarray image: Distorted image.

        :rtype: np.ndarray
        """
        return cv2.undistort(image, self._camera_matrix, self._dist_coefficients)
