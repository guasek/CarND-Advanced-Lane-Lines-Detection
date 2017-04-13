"""
Line class definition.
"""

import numpy as np

y_meters_per_pixel = 30 / 720
x_meters_per_pixel = 3.7 / 700


class LaneLine:
    """
    Single lane line representation
    """

    def __init__(self, sliding_windows_nr, margin, recenter_threshold):

        self._sliding_windows_nr = sliding_windows_nr
        self._margin = margin
        self._recenter_threshold = recenter_threshold
        self._line_detected = False

        self._smooth_iterations_nb = 10
        self._current_polynomial_fit = [0, 0, 0]
        self._polynomial_coefficients_history = []
        self._smoothed_polynomial = [0, 0, 0]

        self.allx = None
        self.ally = None

    def find_in(self, binary_warped_image, lane_current_x_position):
        """
        Looks for a lane lines in a binary warped image.

        :param np.ndarray binary_warped_image    : Binary perspective warped image.
        :param int        lane_current_x_position: Currently found lane x position.
        """
        if self._line_detected:
            self._quick_detect(binary_warped_image)
            return

        nonzero = binary_warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_indices = []
        window_height = np.int(binary_warped_image.shape[0] / self._sliding_windows_nr)
        for window in range(self._sliding_windows_nr):
            lower_bound = binary_warped_image.shape[0] - (window + 1) * window_height
            upper_bound = binary_warped_image.shape[0] - window * window_height
            left_bound = lane_current_x_position - self._margin
            right_bound = lane_current_x_position + self._margin

            indices = (
                (nonzeroy >= lower_bound) & (nonzeroy < upper_bound) &
                (nonzerox >= left_bound) & (nonzerox < right_bound)
            ).nonzero()[0]

            lane_indices.append(indices)
            if len(indices) > self._recenter_threshold:
                lane_current_x_position = np.int(np.mean(nonzerox[indices]))

        lane_indices = np.concatenate(lane_indices)
        x_points = nonzerox[lane_indices]
        y_points = nonzeroy[lane_indices]
        self.fit_polynomial(x_points, y_points)
        self._line_detected = True

    def _quick_detect(self, binary_warped_image):
        """
        Does a quicker search of lane lines basing on previously found fit.

        :param np.ndarray binary_warped_image: Warped binarized image.
        """
        nonzero = binary_warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_indices = (
            (nonzerox > (self.generate_x_points(nonzeroy) - margin)) &
            (nonzerox < (self.generate_x_points(nonzeroy) + margin))
        )

        x_points = nonzerox[lane_indices]
        y_points = nonzeroy[lane_indices]
        self.fit_polynomial(x_points, y_points)

    def fit_polynomial(self, x_points, y_points):
        """
        Finds a polynomial that fits given points the best.

        :param np.ndarray x_points: List of x points,
        :param np.ndarray y_points: List of y points.
        """
        if x_points.size == 0 or y_points.size == 0:
            return
        self._polynomial_coefficients_history.append(np.polyfit(y_points, x_points, 2))
        if len(self._polynomial_coefficients_history) > self._smooth_iterations_nb:
            self._polynomial_coefficients_history.pop(0)

        self.allx = x_points
        self.ally = y_points

    def current_polynomial_fit(self):
        """
        Returns a current smoothed polynomial coefficients.

        :rtype: list
        """
        stacked_data = np.vstack(self._polynomial_coefficients_history)
        smoothed_polynomial = [0, 0, 0]
        smoothed_polynomial[0] = np.average(stacked_data[:, 0])
        smoothed_polynomial[1] = np.average(stacked_data[:, 1])
        smoothed_polynomial[2] = np.average(stacked_data[:, 2])
        return smoothed_polynomial

    def discard_latest_fit(self):
        """
        Discards latest fit
        """
        if len(self._polynomial_coefficients_history) > 3:
            self._polynomial_coefficients_history.pop()
        self._line_detected = False

    def calculate_line_curvature(self):
        """
        Calculates a curvature of the line.

        :rtype: int
        """

        real_world_metric_polynomial = np.polyfit(self.ally * y_meters_per_pixel, self.allx * x_meters_per_pixel, 2)
        return ((1 + (2 * real_world_metric_polynomial[0] * 720 * y_meters_per_pixel +
                      real_world_metric_polynomial[1]) ** 2) ** 1.5) / np.absolute(2 * real_world_metric_polynomial[0])

    def calculate_distance_from_center(self, other_lane_line):
        """
        Calculates distance from center of current and another lane line.

        :param LaneLine other_lane_line: Second lane line.

        :rtype: int
        """
        first_x_position = self.generate_x_points(720)
        second_x_position = other_lane_line.generate_x_points(720)
        if second_x_position < first_x_position:
            first_x_position, second_x_position = second_x_position, first_x_position
        return ((first_x_position + ((second_x_position - first_x_position) / 2)) - 640) * x_meters_per_pixel

    def generate_x_points(self, y_points):
        """
        Generates x points values for given y_points.

        :param y_points: Y points.

        :rtype:
        """
        stacked_data = np.vstack(self._polynomial_coefficients_history)
        smoothed_polynomial = [0, 0, 0]
        smoothed_polynomial[0] = np.average(stacked_data[:, 0])
        smoothed_polynomial[1] = np.average(stacked_data[:, 1])
        smoothed_polynomial[2] = np.average(stacked_data[:, 2])
        return smoothed_polynomial[0] * y_points ** 2 + smoothed_polynomial[1] * y_points + smoothed_polynomial[2]
