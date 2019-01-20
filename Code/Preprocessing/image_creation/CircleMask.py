import copy
import csv
import os
import random
from functools import reduce

import cv2
import numpy as np

import Code.Libraries.MyOculusCsvLib as mocl
import Code.Libraries.MyOculusImageLib as moil
import Code.Libraries.MyOculusRepoNav as morn


# PASTE TO PATIENTS REPO CSV FILE FROM Images/SharedMaskData and commit it modified as another version in SharedMaskData in case
# something went wrong

# Creates mask for random image in repo with specific target size (easy to change with csv recreation)
class CircleMask:

    def __init__(self, image_path, winname, size_level, scale):
        self.image_path = image_path
        self.winname = winname
        self.accepted = True
        self.masks_done = 0
        self.mask = None
        self.rr = 0
        self.xx = 0
        self.yy = 0

        self.targetSize = moil.getColsRows(size_level, scale)

    # mouse callback
    def draw_circle(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.accepted = False
            self.xx = x
            self.yy = y
            mask = np.zeros((self.targetSize[1], self.targetSize[0]), dtype=np.uint8)
            mask = cv2.circle(mask, (self.xx, self.yy), self.rr, 255, -1)
            cv2.imshow('mask', mask)

        if event == cv2.EVENT_RBUTTONDOWN:
            self.accepted = False
            cv2.destroyWindow('a')
            cv2.destroyWindow('mask')

        if event == 10:
            self.accepted = False
            if flags > 0:
                self.rr += 1

            else:
                self.rr -= 1
            self.mask = np.zeros((self.targetSize[1], self.targetSize[0]), dtype=np.uint8)
            self.mask = cv2.circle(self.mask, (self.xx, self.yy), self.rr, 255, -1)
            cv2.imshow('mask', self.mask)

    def circle_mask_on_random_image_in_path(self, path, target_size=None, r=None, extension=".jpg", check_csv=True,
                                            list=None):

        numb = len([i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))])
        temp = ([a for a in os.listdir(path) if os.path.isfile(os.path.join(path, a))])
        try:
            j = np.random.randint(numb)
        except:
            print(path + ", numb: " + str(numb))
            return

        ImName = random.choice(temp)
        if not os.path.exists(path + '/mask'):
            os.makedirs(path + '/mask')
        tempName = path + '/mask/' + ImName
        if os.path.exists(tempName):
            print("Path exists (" + tempName + ")")
            return
        if check_csv:
            paths = morn.getRepoPathAndImagePath(path)
            row = paths[1].split("/")[:-1]
            row.append(ImName)
            if mocl.checkIfExistsInCSV(row, paths[0], list, False):
                print("In CSV exists (" + tempName + ")")
                return

        if r is None and target_size is not None:
            self.rr = int(target_size[0] / 10)
        else:
            self.rr = r

        img = moil.read_and_size(ImName, path=path, target_size=target_size, extension='')
        w, h, c = moil.getWidthHeightChannels(img)
        if r is None and target_size is None:
            self.rr = int(w / 10)
            target_size = (w, h)
        moil.show(img)

        accepted = False
        while not accepted:
            accepted = True

            im2, contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im2 = copy.deepcopy(img)
            cv2.drawContours(im2, contours, 0, (0, 255, 255), 2)

            moil.show(im2)

        split_path = path.split("/")[:-1]

        repo_path = reduce((lambda x, y: x + '/' + y), split_path[:len(split_path) - 3])
        if not os.path.isfile(repo_path + "/maskData.csv"):
            csvFile = open(repo_path + '/maskData.csv', 'w', newline="")
            writer = csv.writer(csvFile)
            writer.writerow(['patient', 'date', 'eye', 'name', 'width', 'height', 'x', 'y', 'r'])
            csvFile.close()

        csvFile = open(repo_path + '/maskData.csv', 'a', newline="")
        writer = csv.writer(csvFile)
        ls = split_path[-3:]
        ls.extend([ImName, target_size[0], target_size[1], self.xx, self.yy, self.rr])
        writer.writerow(ls)
        csvFile.close()
        cv2.imwrite(path + '/mask/' + ImName, self.mask)
        self.masks_done += 1
        print("masks: " + str(self.masks_done))
        cv2.destroyWindow('mask')

    def CreateMasks(self):
        cv2.destroyWindow(self.winname)
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, self.draw_circle)
        list = mocl.getCsvList(self.image_path, False)
        while (True):
            self.circle_mask_on_random_image_in_path(morn.random_path(self.image_path), target_size=self.targetSize,
                                                     check_csv=True,
                                                     list=list)


if __name__ == "__main__":
    level = 30
    base_scale = 0.75

    path = '../../../Images/all/'

    obj = CircleMask(path, 'win', level, base_scale)
    obj.CreateMasks()
