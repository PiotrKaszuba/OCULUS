import csv
import math
import shutil
from functools import reduce

import cv2
import numpy as np

import Code.Libraries.MyOculusCsvLib as mocl
import Code.Libraries.MyOculusRepoNav as morn


# Creates masks for image shape in new repo based on imageData and maskData in repo (for every image search if there was mask)
class BuildMasksRepo:

    def __init__(self, repo_base, repo_name, new_name):
        self.old_repo = repo_base + repo_name + '/'
        self.new_repo = repo_base + new_name + '/'

    def createMaskFromCsv(self, repo_path, imageRow, list=None, override=False):
        if list is None:
            list = mocl.getCsvList(repo_path, image=False)

        target = mocl.checkIfExistsInCSV(imageRow, repo_path=repo_path, list=list, image=False, returnTargetRow=True)
        if target is None:
            return False

        imageW = int(imageRow[4])
        imageH = int(imageRow[5])

        maskW = int(target[4])
        maskH = int(target[5])
        maskX = int(target[6])
        maskY = int(target[7])
        maskR = int(target[8])

        Wratio = imageW / maskW
        Hratio = imageH / maskH

        Rratio = math.sqrt((((Wratio ** 2) + (Hratio ** 2)) / 2))

        outX = int(maskX * (Wratio))
        outY = int(maskY * (Hratio))
        outR = int(maskR * (Rratio))
        mask = np.zeros((imageH, imageW), dtype=np.uint8)
        mask = cv2.circle(mask, (outX, outY), outR, 255, -1)

        path = repo_path + reduce((lambda x, y: x + '/' + y), imageRow[:3]) + '/mask/'

        return morn.createImageInPath(path, imageRow[3], mask, override)

    def createAllMasksForImagesCsv(self, repo_path):
        success = 0
        fail = 0
        list = mocl.getCsvList(repo_path, image=False)
        iter = 0
        with open(repo_path + "imageData.csv", 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if self.createMaskFromCsv(repo_path, row, list, True):
                    success += 1
                else:
                    fail += 1
                iter += 1
                print("Images looped: " + str(iter))
            file.close()
        print("Masks created: " + str(success) + ", failed to create: " + str(fail))

    def copyMaskData(self, old_repo_path, new_repo_path):
        shutil.copyfile(old_repo_path + 'maskData.csv', new_repo_path + 'maskData.csv')

    def Build(self):
        self.copyMaskData(self.old_repo, self.new_repo)
        self.createAllMasksForImagesCsv(self.new_repo)


if __name__ == "__main__":
    repo_base = "../../../Images/"
    repo_name = "all"
    new_name = "validateRepo"

    obj = BuildMasksRepo(repo_base, repo_name, new_name)
    obj.Build()
