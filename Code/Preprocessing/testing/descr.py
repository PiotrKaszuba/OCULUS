import Code.Libraries.MyOculusLib as mol

path = '../../../Images/all/'
mol.init(im_path=path, mouse_f=mol.draw_elipse)

mol.all_path(mol.get_description_full_path,  once_per_date=True, dict=True)
