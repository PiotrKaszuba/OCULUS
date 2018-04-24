import Code.Libraries.MyOculusLib as mol
from Code.Preprocessing import Gradients,LBP
import numpy as np
import cv2
import os
class MergeChannels:
    def __init__(self, equalize=False, testMode=False):
        self.Grad = Gradients.Gradients(equalize=False)
        self.LBP = LBP.LocalBinaryPatterns(40, 3.5, method='default', equalize=True, negative=True)
        self.equalize=equalize
        self.testMode=testMode

    def Merge(self, im):
        h,w=np.shape(im)
        merged = np.zeros((h, w, 3))
        lbp =  self.LBP.describe(im)
        grad = self.Grad.getGradientSum(im)
        if self.equalize:
            im = cv2.equalizeHist(im)
            lbp = cv2.equalizeHist(lbp)
            grad = cv2.equalizeHist(grad)
        if self.testMode:
            mol.show(im, other_im=[lbp, grad])
        merged[:, :, 0] = im
        merged[:,:,1] = lbp
        merged[:,:,2] = grad
        return merged.astype('uint8')

    def MergeOnPath(self, path):
        for i in range(len(os.listdir(path)) - 1):
            img = mol.read_and_size(str(i), path=path)
            merged = self.Merge(img)
            mol.show(merged)

