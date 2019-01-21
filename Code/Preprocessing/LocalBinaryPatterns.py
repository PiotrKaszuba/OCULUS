import os

import cv2
from skimage import feature

import Code.Libraries.MyOculusImageLib as moil


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
            moil.show(lbp, other_im=[im])
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

        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method=self.method).astype("uint8")
        # (hist, _) = np.histogram(lbp.ravel(),
        #  bins=np.arange(0, self.numPoints + 3),
        # range=(0, self.numPoints + 2))

        # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        if self.equalize:
            lbp = cv2.equalizeHist(lbp)
        if self.negative:
            lbp = cv2.bitwise_not(lbp)
        return lbp
