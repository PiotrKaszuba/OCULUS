import Code.Libraries.MyOculusLib as mol

path = '../../../Images/repoTest/'

mol.init(im_path=path, mouse_f=mol.draw_circle)

while(True):
    mol.circle_mask_on_random_image_in_path(mol.random_path(path), target_size=(480,384))
