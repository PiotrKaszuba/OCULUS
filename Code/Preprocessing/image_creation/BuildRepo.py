import Code.Libraries.MyOculusRepoNav as morn
import Code.Libraries.MyOculusCsvLib as mocl
import Code.Libraries.MyOculusImageLib as moil
import Code.Preprocessing.MergeChannels as mc
import Code.Algorithms.Models as md
import os
from functools import reduce

#Builds a Repo based on existing repo (only images with mask) changing target image size and preprocessing images with function
class BuildRepo:

    def __init__(self, repo_base, repo_name, new_name, size_level, scale=0.75, function= lambda x:x, onlyMasked=True, override=True):
        self.old_repo = repo_base + repo_name + '/'
        self.new_repo = repo_base + new_name + '/'
        self.target_size = moil.getColsRows(size_level, scale)
        self.onlyMasked = onlyMasked
        self.override = override
        self.function = function

    @staticmethod
    def createImagesRepo(base_repo_path, repo_name):
        if not os.path.exists(base_repo_path + repo_name):
            os.makedirs(base_repo_path + repo_name)
            return True
        return False

    @staticmethod
    def getBaseRepoPathAndRepoName(repo_path):
        repo_path = repo_path.split('/')[:-1]
        base = reduce((lambda x, y: x + '/' + y), repo_path[:len(repo_path) - 1])
        name = repo_path[-1]
        return base, name

    def createImagesInRepoAfterFunctionOnPath(self, path, new_repo_path, function, target_size, override=False,
                                              extension='.jpg', onlyMasked=False):
        success = 0
        fail = 0
        repo_path, image_path = morn.getRepoPathAndImagePath(path)

        base2, name2 = BuildRepo.getBaseRepoPathAndRepoName(new_repo_path)
        BuildRepo.createImagesRepo(base2 + '/', name2)
        new_path = new_repo_path + image_path

        maskList = None
        if onlyMasked:
            maskList = mocl.getCsvList(repo_path, False)

        for a in os.listdir(path):
            if not os.path.isfile(os.path.join(path, a)):
                continue
            patient, date, eye = morn.getPatientDateEye(image_path)

            if onlyMasked and not os.path.isfile(os.path.join(path + 'mask/', a)) and not mocl.checkIfExistsInCSV(
                    [patient, date, eye, a], list=maskList, image=False):
                continue
            name = a.split(".")[0]
            base_image = moil.read_and_size(name, path=path, target_size=target_size)
            image = function(base_image)

            if morn.createImageInPath(new_path, name + extension, image, override):
                mocl.registerImageCsv(new_repo_path, image_path, name + extension, image, function)
                success += 1
            else:
                fail += 1
        return success, fail


    def createFromAllPathImageAfterFunction(self, old_repo_path, new_repo_path, function, target_size=None, eye=None,
                                            override=False, extension='.jpg', onlyMasked=False):
        success = 0
        fail = 0
        if eye != 'left' and eye != 'right':
            eye = 'both'

        patient = None

        iter = 0
        patient = os.listdir(old_repo_path)
        for i in range(len([a for a in patient if not os.path.isfile(os.path.join(old_repo_path, a))])):

            date = os.listdir(old_repo_path + patient[i] + "/")

            for j in range(len(date)):
                if eye == 'left':
                    t1, t2 = self.createImagesInRepoAfterFunctionOnPath(
                        old_repo_path + patient[i] + '/' + date[j] + '/' + 'left_eye_images/', new_repo_path, function,
                        target_size, override, extension, onlyMasked)
                    success += t1
                    fail += t2
                if eye == 'right':
                    t1, t2 = self.createImagesInRepoAfterFunctionOnPath(
                        old_repo_path + patient[i] + '/' + date[j] + '/' + 'right_eye_images/', new_repo_path, function,
                        target_size, override, extension, onlyMasked)
                    success += t1
                    fail += t2
                if eye == 'both':
                    t1, t2 = self.createImagesInRepoAfterFunctionOnPath(
                        old_repo_path + patient[i] + '/' + date[j] + '/' + 'left_eye_images/', new_repo_path, function,
                        target_size, override, extension, onlyMasked)
                    success += t1
                    fail += t2
                    t1, t2 = self.createImagesInRepoAfterFunctionOnPath(
                        old_repo_path + patient[i] + '/' + date[j] + '/' + 'right_eye_images/', new_repo_path, function,
                        target_size, override, extension, onlyMasked)
                    success += t1
                    fail += t2
            iter += 1
            print("Patients finished: " + str(iter))

        print("Images created: " + str(success) + ", attempts failed to create: " + str(fail))
    def Build(self):
        self.createFromAllPathImageAfterFunction(self.old_repo, self.new_repo, self.function, target_size=self.target_size, onlyMasked=self.onlyMasked, override=self.override)

if __name__ == "__main__":
    level = 20
    scale = 0.75

    equalize = True
    mer= mc.MergeChannels(equalize)

    repo_base = "../../../Images/"
    repo_name = "all"
    new_name = "tarczaGray500"

    onlyMasked = True
    override = True

    obj = BuildRepo(repo_base, repo_name, new_name, level, scale, mer.Merge, onlyMasked, override)
    obj.Build()
