"""
Lane detector class definition
"""
import cv2
import numpy as np

from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from lane_line import LaneLine


class LaneDetector:
    """
    Detects lane lines.

    :param Camera camera: Calibrated camera responsible for removing distortions in images.
    """

    def __init__(self, camera):
        """
        Initializer.
        """
        self._camera = camera
        source_points = np.float32([[170, 720], [566, 450], [716, 450], [1120, 720]])
        dest_points = np.float32([[350, 720], [200, 0], [1080, 0], [930, 720]])

        self._transformation_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        self._inverse_transformation_matrix = cv2.getPerspectiveTransform(dest_points, source_points)

        sliding_windows_nr = 9
        margin = 100
        recenter_threshold = 50
        self._left_lane_line = LaneLine(sliding_windows_nr, margin, recenter_threshold)
        self._right_lane_line = LaneLine(sliding_windows_nr, margin, recenter_threshold)

    def detect_lane_lines_in_video(self, video_path, output_file_name):
        """
        Detects lane lines in an entire video clip.

        :param str video_path      : Path to a source video file.
        :param str output_file_name: Name of the output file.
        """
        source_clip = VideoFileClip(video_path)
        converted_clip = source_clip.fl_image(self.detect_lane_lines_in_image)
        converted_clip.write_videofile(output_file_name, audio=False)

    def detect_lane_lines_in_image(self, image):
        """
        Detects lane lines in single image,

        :param np.ndarray image: An image to detect lines on.

        :rtype: np.ndarray
        """
        undistorted_image = self._camera.correct_distortions(image)
        thresholded_image = self.threshold(undistorted_image)
        birdseye_view_image = self.transform_perspective(thresholded_image)
        self.update_lane_lines_positions(birdseye_view_image)
        lanes_detected = self.draw_lanes_on_image(undistorted_image, birdseye_view_image)
        return lanes_detected

    def correct_distortions(self, image):
        """
        Corrects image distortions.

        :param np.ndarray image: Image with distortions.

        :rtype: np.ndarray
        """
        return self._camera.correct_distortions(image)

    def threshold(self, image):
        """
        Thresholds an image in order to find lines.

        :param np.ndarray image: Image to put thresholds on.

        :rtype: np.ndarray
        """

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)

        s_channel = hls[:, :, 2]
        s_thresh = (130, 255)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        l_channel = hls[:, :, 1]
        sobel_on_l_thresh = (40, 255)
        sobel_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_l = np.absolute(sobel_l)
        scaled_sobel_l = np.uint8(255 * abs_sobel_l / np.max(abs_sobel_l))
        sobel_on_l_binary = np.zeros_like(scaled_sobel_l)
        sobel_on_l_binary[(scaled_sobel_l >= sobel_on_l_thresh[0]) & (scaled_sobel_l <= sobel_on_l_thresh[1])] = 1

        sobel_s_thresh = (30, 255)
        sobel_s = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_s = np.absolute(sobel_s)
        scaled_sobel_s = np.uint8(255 * abs_sobel_s / np.max(abs_sobel_s))
        sobel_on_s_binary = np.zeros_like(scaled_sobel_s)
        sobel_on_s_binary[(scaled_sobel_s >= sobel_s_thresh[0]) & (scaled_sobel_s <= sobel_s_thresh[1])] = 1

        combined = np.zeros_like(s_binary)
        combined[((sobel_on_l_binary == 1) | ((sobel_on_s_binary == 1) & (s_binary == 1)))] = 255
        return combined

    def transform_perspective(self, image):
        """
        Transforms image perspective.

        :param np.ndarray image: Image to transform.

        :rtype: np.ndarray
        """
        warped = cv2.warpPerspective(image, self._transformation_matrix, (1200, 720), flags=cv2.INTER_LINEAR)
        return warped

    def update_lane_lines_positions(self, binary_warped_image):
        """
        Looks for lane lines in a binary warped image.

        :param np.ndarray binary_warped_image: Binary warped image
        """
        histogram = np.sum(binary_warped_image[binary_warped_image.shape[0] / 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)

        left_lane_base_x_position = np.argmax(histogram[:midpoint])
        self._left_lane_line.find_in(binary_warped_image, left_lane_base_x_position)

        right_lane_base_x_position = np.argmax(histogram[midpoint:]) + midpoint
        self._right_lane_line.find_in(binary_warped_image, right_lane_base_x_position)

        left_current_fit = self._left_lane_line.current_polynomial_fit()
        right_current_fit = self._right_lane_line.current_polynomial_fit()
        C_difference = left_current_fit[2] - right_current_fit[2]
        B_difference = left_current_fit[1] - right_current_fit[1]
        A_difference = left_current_fit[0] - right_current_fit[0]
        if not (
            -655 < C_difference < -400 and
            -0.29 < B_difference < 0.3 and
            -2.33958539e-04 < A_difference < 1.59358225e-04
        ):
            self._left_lane_line.discard_latest_fit()
            self._right_lane_line.discard_latest_fit()

    def draw_lanes_on_image(self, image, warped_image):
        """
        Draws lane lines on an image.

        :param np.ndarray image       : Image to draw on
        :param np.ndarray warped_image: Warped perspective image.

        :rtype: np.ndarray
        """
        y_points = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
        left_x_points = self._left_lane_line.generate_x_points(y_points)
        right_x_points = self._right_lane_line.generate_x_points(y_points)

        left_lane_points = np.array([np.transpose(np.vstack([left_x_points, y_points]))])
        right_lane_points = np.array([np.flipud(np.transpose(np.vstack([right_x_points, y_points])))])
        combined_points = np.hstack((left_lane_points, right_lane_points))

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        warped_drawn_lane = np.dstack((warp_zero, warp_zero, warp_zero))
        cv2.fillPoly(warped_drawn_lane, np.int_([combined_points]), (0, 255, 0))

        original_perspective_drawn_lane = cv2.warpPerspective(
            warped_drawn_lane, self._inverse_transformation_matrix, (image.shape[1], image.shape[0])
        )

        result = cv2.addWeighted(image.astype(np.uint8), 1, original_perspective_drawn_lane, 0.3, 0)

        curvature_text = 'Radius of curvature: {:.2f} m'.format(self._left_lane_line.calculate_line_curvature())
        distance_from_center_text = 'Distance from center: {:.2f} m'.format(
            self._left_lane_line.calculate_distance_from_center(self._right_lane_line)
        )

        cv2.putText(result, curvature_text, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
        cv2.putText(result, distance_from_center_text, (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
        return result
