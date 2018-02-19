import MyOculusLib as mol

mol.init()

mol.all_path(lambda x : mol.show(mol.read_and_size("0", scale=0, h_div_w=0.75, not_modified_wanted_value=300,  path = x)) )