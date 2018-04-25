import Code.Libraries.MyOculusLib as mol
import Code.Algorithms.Models as md
path = '../../../Images/repoTest/'

mol.init(im_path=path, mouse_f=mol.draw_circle)

level = 30
base_scale = 0.75

targetSize = md.Models.getColsRows(level, base_scale)

while(True):
    mol.circle_mask_on_random_image_in_path(mol.random_path(path), target_size=targetSize)
