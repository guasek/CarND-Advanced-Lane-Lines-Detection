"""
Entry point for an application
"""

import gallery

from camera import Camera
from lane_detector import LaneDetector


calibration_images_dir = '../camera_cal/'
test_images_dir = '../test_images/'

project_video = '../project_video.mp4'
project_video_output = '../project_video_output.mp4'

if __name__ == "__main__":
    camera = Camera.calibrate(calibration_images_dir, 9, 6)
    lane_detector = LaneDetector(camera)
    gallery.show_test_images_pipeline(camera, test_images_dir)
    lane_detector.detect_lane_lines_in_video(project_video, project_video_output)
