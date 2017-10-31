import numpy as np
import cv2


class faceDetection:
    def __init__(self):
        self.haar_cascades = list()
        self.haar_cascades.append(cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml'))
        self.haar_cascades.append(cv2.CascadeClassifier('../data/haarcascade_lefteye_2splits.xml'))
        self.haar_cascades.append(cv2.CascadeClassifier('../data/haarcascade_righteye_2splits.xml'))

    def getRect(self, xywh):
        return xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]

    def drawRect(self, img, xyhw, color=(0,255,0)):
        x1, y1, x2, y2 = self.getRect(xyhw)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        return

    def getRoi(self, img, xywh):
        return img[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]]

    # TODO : Resizing eye bbs
    # def resize_eyes(locations):
    #     max_hw = max(lw, lh, rw, rh)
    #     return 0

    def detect(self, img, DRAW=0, n_repeats=1):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            face = self.haar_cascades[0].detectMultiScale(gray, 1.3, 5)[0,:]
        except:
            return None

        eye_roi = [face.copy(), face.copy()]
        eye_roi[0][2] = face[2]/2
        eye_roi[1][2] = face[2]/2
        eye_roi[1][0] = face[0] + face[2]/2

        eyes = list()
        for i in range(2):
            try:
                eye = self.haar_cascades[1+i].detectMultiScale(self.getRoi(gray, eye_roi[i]))[0,:]
            except:
                return None
            for j in range(2):
                eye[j] += eye_roi[i][j]
            eyes.append(eye)
        del [eye]

        face_eyes = [face] + eyes

        if DRAW>0:
            draw_img = img.copy()
            if DRAW>1:
                for i in range(3):
                    self.drawRect(draw_img, face_eyes[i])

        face_eyes[0][2:] = max(face_eyes[0][2:])

        # resize_eyes(...)
        eyes_dim = np.max(np.concatenate((face_eyes[1][2:], face_eyes[2][2:])))
        for i in range(2):
            face_eyes[1+i][2:] = eyes_dim

        # Draw again
        if DRAW>0:
            for i in range(3):
                self.drawRect(draw_img, face_eyes[i], color=(255,0,0))
            cv2.imshow('image',draw_img)
            cv2.waitKey(10)

        # Generate face grid
        grid_face = np.floor(face * 25 / 720).astype(int)
        facegrid = np.zeros((25,25))
        x1, y1, x2, y2 = self.getRect(grid_face)
        facegrid[y1:y2, x1:x2] = 1

        # Rescale images
        images = list()
        for i in range(3):
            thisImage = self.getRoi(img, face_eyes[i])
            thisImage = cv2.resize(thisImage, (224,224))
            images.append(thisImage)

            if DRAW>1:
                cv2.imshow('image',thisImage)
                cv2.waitKey(10)

        facegrid = np.reshape(facegrid,(1,625,1,1))
        images.append(facegrid)

        return images
