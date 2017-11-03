import cv2
import faceDetection
import visualizeGaze
import iTracker
import numpy as np
import sys

MOVING_AVG = 3


def liveGazeCapture(config):
    # Initialize iTracker and face detection
    itrack = iTracker.iTracker()
    detector = faceDetection.faceDetection()

    # Initialize cam
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cv2.namedWindow("image")
    cv2.imshow("image",img)
    cv2.waitKey(500)

    # Initialize reslt visualization
    vis = visualizeGaze.visualize(config)

    # Initialize MA filter
    coords_array = np.zeros((3,2))
    ma_counter = 0

    while True:

        ret, img = cap.read()
        img = img[:,80:510,:]

        images = detector.detect(img, DRAW=0)
        if images is None:
            continue

        coords = itrack.infer(images)  # x, y in cm

        coords_array[ma_counter, :] = coords
        coords_filtered = np.mean(coords_array, axis=0)
        if ma_counter < MOVING_AVG - 1:
            ma_counter += 1
        else:
            ma_counter = 0

        vis.showGaze(coords_filtered)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = map(int, sys.argv[1])[0]
    else:
        config = 0
    liveGazeCapture(config)
