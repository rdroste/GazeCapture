import caffe
import numpy as np
import cv2

model = './models/itracker_deploy.prototxt'
weights = './models/snapshots/itracker25x_iter_92000.caffemodel'

mean_path = "models/"
mean_image_files = ["mean_images/mean_face_224.binaryproto",
                    "mean_images/mean_right_224.binaryproto",
                    "mean_images/mean_left_224.binaryproto"]


class iTracker:
    def __init__(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net(model, weights, caffe.TEST)

        self.mean_image = [self.loadMeanImage(mean_path + mean_image_files[i])
                           for i in range(3)]
        self.n_repeats = 1

    def loadMeanImage(self, fname):
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(fname, 'rb').read()
        blob.ParseFromString(data)
        data = np.array(blob.data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        return arr

    def subtractMeanImages(self, images):
        for i in range(3):
            images[i] = images[i] - self.mean_image[i]

    def reshapeImages(self, images):
        final_images = list()
        for i in range(3):
            thisImage = images[i].copy()
            thisImage = np.reshape(thisImage,(224,224,3,1))
            thisImage = np.repeat(thisImage,self.n_repeats,axis=3)
            thisImage = np.transpose(thisImage, (3,2,0,1))
            final_images.append(thisImage.copy())

        final_images.append(np.reshape(images[3],(1,625,1,1), order='F'))
        return final_images

    def infer(self, images):
        final_images = self.reshapeImages(images)
        self.subtractMeanImages(final_images)
        inf = self.net.forward_all(**{"image_left":final_images[2], "image_right":final_images[1],
            "image_face":final_images[0], "facegrid":final_images[3]})
        coords = inf['fc3'][0]
        return coords
