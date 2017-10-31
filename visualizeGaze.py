import numpy as np
import cv2


class visualize:
    def __init__(self, config):
        if config is 0:
            screen_res = np.array([1440, 2560])
            ul_corner = np.array([0,100])
            lr_corner = np.array([100,100])
            ppi = 210  # 210 / 2.54
            header_cm = 1
            cam2screen_cm = 1.3

        self.reducedRes = screen_res - ul_corner - lr_corner
        self.ppcm = ppi/2.54
        self.y_offset_px = (cam2screen_cm + header_cm) * self.ppcm

        cv2.namedWindow("gaze")
        self.resetGazeMap()
        self.showGazeMap()
        cv2.moveWindow("gaze", ul_corner[1], ul_corner[0])
        cv2.waitKey(500)

    def resetGazeMap(self):
        self.gazeMap = np.zeros(tuple(np.append(self.reducedRes,3)), np.uint8)
        cv2.circle(self.gazeMap, tuple(np.flip(self.reducedRes,0)/2), 5, (255,255,255), -1)

    def showGazeMap(self):
        cv2.imshow("gaze", self.gazeMap)
        cv2.waitKey(5)

    def newVis(self, coords):
        coords[1] = -coords[1]
        coords = np.round(coords * self.ppcm + np.array([self.reducedRes[1]/2, -self.y_offset_px])).astype(int)
        print(coords)
        self.resetGazeMap()
        cv2.circle(self.gazeMap, tuple(coords), 5, (0,0,255), -1)
        self.showGazeMap()
