import random
import shutil
import os
import Code.Preprocessing.image_creation.BuildMasksRepo as bmr
import Code.Preprocessing.image_creation.BuildRepo as br
import Code.Preprocessing.MergeChannels as mc
if __name__ == "__main__":
    level = 20
    scale = 0.75

    equalize = True
    mer = mc.MergeChannels(equalize)

    repo_base = "../Images/"
    repo_name = "all"
    new_name = "TarczaSAB500"

    shared_mask_data_filename = 'maskDataTrain500.csv'

    if os.path.exists(repo_base + repo_name + '/maskData.csv'):
        shutil.copyfile(repo_base + repo_name + '/maskData.csv',
                        repo_base + repo_name + '/maskData_copyFile' + str(random.randint(0, 10000))+'.csv')
    shutil.copyfile("../SharedMaskData/" + shared_mask_data_filename,
                    repo_base + repo_name + '/maskData.csv')
    onlyMasked = True
    override = True

    obj = br.BuildRepo(repo_base, repo_name, new_name, level, scale, mer.Merge, onlyMasked, override)
    obj.Build()

    obj = bmr.BuildMasksRepo(repo_base, repo_name, new_name)
    obj.Build()
