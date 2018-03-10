import MyOculusLib as mol


path = 'Images/all/'
path_to_patients = 'Images/patients/'
filename= 'Pio!!tr'

mol.delete_directories(mol.same_lines(filename,path_to_txt=path_to_patients, path_to_local=path),path)