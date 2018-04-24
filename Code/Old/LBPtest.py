import cv2
import numpy as np
import os
import Code.Libraries.MyOculusLib as mol
def LBP(image):
    newimage = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            temp = LBP_Mask(image, i - 1, j - 1)
            temp2 = []
            temp2.extend(temp[0, :])
            temp2.append(temp[1, 2])
            temp2.extend(list(reversed(temp[2, :])))
            temp2.append(temp[1, 0])
            temp2 = np.asarray(temp2)
            temp2 = np.roll(temp2, 0)
            newpixel = 0
            for g in range(temp2.size):
                if temp2[g] >= image[i, j]:
                    newpixel += 2 ** (temp2.size - g - 1)
            newimage[i, j] = newpixel

    return newimage


def LBP_Mask(image, i, j):
    return image[i:i + 3, j:j + 3]


def scale(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

def lbptestOnPath(path):
        for i in range(len(os.listdir(path)) - 1):
            img = mol.read_and_size(str(i), path=path)
            lbp = LBP(img)
            mol.show(lbp)


