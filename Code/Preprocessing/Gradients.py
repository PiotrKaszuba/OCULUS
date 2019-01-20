import os

import cv2

import Code.Libraries.MyOculusImageLib as moil


class Gradients:
    def __init__(self, kernel=3, equalize=False, depth=5, SobelXweight=0.4, SobelYweight=0.4, LaplacianWeight=0.2):

        self.kernel = kernel
        self.equalize = equalize
        self.depth = depth
        self.SobelXweight = SobelXweight
        self.SobelYweight = SobelYweight
        self.LaplacianWeight = LaplacianWeight

    def GradientSumOnPath(self, path):
        for i in range(len(os.listdir(path)) - 1):
            img = moil.read_and_size(str(i), path=path)
            grad = self.getGradientSum(img)
            moil.show(grad)

    def getGradientSum(self, im):
        if self.equalize:
            im = cv2.equalizeHist(im)
        return cv2.convertScaleAbs(self.SobelXweight * self.getSobelX(im) + self.SobelYweight * self.getSobelY(
            im) + self.LaplacianWeight * self.getLaplacian(im))

    def getSobelX(self, im):
        return cv2.Sobel(im, self.depth, 1, 0, ksize=self.kernel)

    def getSobelY(self, im):
        return cv2.Sobel(im, self.depth, 0, 1, ksize=self.kernel)

    def getLaplacian(self, im):
        return cv2.Laplacian(im, self.depth, ksize=self.kernel)
