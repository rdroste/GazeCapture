import numpy as np
import cv2


class visualize:
    def __init__(self, config):

        # deault config
        screen_res = np.array([1440, 2560])
        ul_corner = np.array([0,100])
        lr_corner = np.array([100,100])
        ppi = 210  # 210 / 2.54
        header_cm = 1
        cam2screen_cm = 1.3
        self.ppcm = ppi/2.54
        self.reducedRes = screen_res - ul_corner - lr_corner
        self.y_offset_px = (cam2screen_cm + header_cm) * self.ppcm
        self.pos_scaling = np.array([0.15, 0.5])

        if config == 1:
            self.y_offset_px = - self.reducedRes[0]/2
            self.pos_scaling = np.array([0.5, 0.5])

        cv2.namedWindow("gaze")
        self.resetGazeMap()
        self.showGazeMap()
        cv2.moveWindow("gaze", ul_corner[1], ul_corner[0])
        cv2.waitKey(500)

    def resetGazeMap(self):
        self.gazeMap = np.zeros(tuple(np.append(self.reducedRes,3)), np.uint8)
        point_pos = tuple(np.flip(self.reducedRes*self.pos_scaling,0).astype(np.int))
        cv2.circle(self.gazeMap, point_pos, 5, (255,255,255), -1)

    def showGazeMap(self):
        cv2.imshow("gaze", self.gazeMap)
        cv2.waitKey(5)

    def showGaze(self, coords):
        coords[1] = -coords[1]
        coords = np.round(coords * self.ppcm + np.array([self.reducedRes[1]/2, -self.y_offset_px])).astype(int)
        print(coords)
        self.resetGazeMap()
        cv2.circle(self.gazeMap, tuple(coords), 5, (0,0,255), -1)
        self.showGazeMap()
