from Code.Libraries import MyOculusLib as mol

path = '../../Images/data/'
string = mol.string_patients(path)

filename = 'Piotr'
path_to_patients = '../../Images/patients/'

mol.to_txt(string, filename, path_to_patients)