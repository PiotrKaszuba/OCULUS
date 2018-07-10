import Code.Libraries.MyOculusLib as mol

path = '../../../Images/data/'
mol.init(im_path=path, mouse_f=mol.draw_elipse)

mol.choose_mappings()