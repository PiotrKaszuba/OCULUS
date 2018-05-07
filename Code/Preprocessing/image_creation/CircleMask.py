
#PASTE TO PATIENTS REPO CSV FILE FROM Images/SharedMaskData and commit it modified as another version in SharedMaskData in case
#something went wrong

#Creates mask for random image in repo with specific target size (easy to change with csv recreation)

import Code.Libraries.MyOculusLib as mol
import Code.Algorithms.Models as md
path = '../../../Images/data/'

mol.init(im_path=path, mouse_f=mol.draw_circle)

level = 30
base_scale = 0.75

targetSize = md.Models.getColsRows(level, base_scale)

list = mol.getCsvList(path,False)

while(True):
    mol.circle_mask_on_random_image_in_path(mol.random_path(path), target_size=targetSize, check_csv=True, list=list)
