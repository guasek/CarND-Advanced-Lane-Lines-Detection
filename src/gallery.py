"""
Displays images
"""
import glob
import cv2
import numpy as np
from lane_detector import LaneDetector
from matplotlib import pyplot as plt


def show_chessboards(camera, calibration_images_dir):
    """
    displays undistorted calibration images.
    """

    paths_to_images = glob.glob(calibration_images_dir + "*.jpg")
    for image_path in paths_to_images:

        distorted_image = cv2.imread(image_path)
        undistorted_image = camera.correct_distortions(distorted_image)

        figure, (ax1, ax2) = plt.subplots(1, 2)
        figure.tight_layout()
        ax1.imshow(distorted_image)
        ax1.set_title('Original Image')
        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image')

    plt.show()


def plot_detected_lines(lane_detector, birdseye_view_image):
    """
    Plots detected lines on an image

    :return:
    """

    y_points = np.linspace(0, birdseye_view_image.shape[0] - 1, birdseye_view_image.shape[0])
    left_x_points = lane_detector._left_lane_line.generate_x_points(y_points)
    right_x_points = lane_detector._right_lane_line.generate_x_points(y_points)

    out_img = np.dstack((birdseye_view_image, birdseye_view_image, birdseye_view_image))*255
    window_img = np.zeros_like(out_img)

    left_line_window1 = np.array([np.transpose(np.vstack([left_x_points-100, y_points]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_x_points+100, y_points])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_x_points-100, y_points]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_x_points+100, y_points])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_x_points, y_points, color='yellow')
    plt.plot(right_x_points, y_points, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def show_test_images_pipeline(camera, test_images_dir):
    """
    Shows detection pipeline results step by step.

    :param Camera camera         : Camera object.
    :param str    test_images_dir: Path to directory containing test images.
    """
    paths_to_images = glob.glob(test_images_dir + "*.jpg")
    for image_path in paths_to_images:
        image = cv2.imread(image_path)
        lane_detector = LaneDetector(camera)
        undistorted_image = lane_detector.correct_distortions(image)
        thresholded_image = lane_detector.threshold(undistorted_image)
        birdseye_image = lane_detector.transform_perspective(thresholded_image)
        lane_detector.update_lane_lines_positions(birdseye_image)
        lines_detected = lane_detector.draw_lanes_on_image(undistorted_image, birdseye_image)
        plt.imshow(cv2.cvtColor(lines_detected, cv2.COLOR_BGR2RGB))
        plt.show()
