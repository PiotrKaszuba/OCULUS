import os

import cv2
from skimage import feature

import MyOculusImageLib as moil


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius, method="uniform", equalize=False, negative=False, scale=0):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.method = method
        self.equalize = equalize
        self.negative = negative
        self.scale = scale

    def LbpOnPath(self, path):
        for i in range(len(os.listdir(path)) - 1):
            img = moil.read_and_size(str(i), path=path, scale=self.scale)
            lbp = self.describe(img)
            moil.show(lbp, other_im=[img])
            # (fig, ax) = plt.subplots()
            # ax.hist(lbp.ravel(), normed=True, bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

            # plt.show()

    '''def HistOnPath(self, path):
        for i in range(len(os.listdir(path)) - 1):
            img = mol.read_and_size(str(i), path=path)
            lbp = self.pureLbp(img)
            #mol.show(lbp)
            (fig, ax) = plt.subplots()
            ax.hist(lbp.ravel(), normed=True, bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
            ax.set_ylim([0, 0.030])
            plt.show()'''

    def pureLbp(self, image):
        return feature.local_binary_pattern(image, self.numPoints,
                                            self.radius, method=self.method)

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        #moil.show(image)
        #lbp = cv2.medianBlur(image, ksize=3)
        #moil.show(lbp)
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method=self.method)
        # (hist, _) = np.histogram(lbp.ravel(),
        #  bins=np.arange(0, self.numPoints + 3),
        # range=(0, self.numPoints + 2))

        # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + eps)
        #moil.show(lbp)
        # return the histogram of Local Binary Patterns

        import numpy as np
        #lbp = lbp / np.iinfo(np.uint64).max
        #lbp = lbp * 255
        #lbp = lbp.astype("uint8")
        unique, counts = np.unique(lbp, return_counts=True)
        num = len(unique)
        lbp = lbp.astype('uint8')

        if self.negative:
            lbp = cv2.bitwise_not(lbp)
        if self.equalize:
            lbp = cv2.equalizeHist(lbp)
        return lbp
