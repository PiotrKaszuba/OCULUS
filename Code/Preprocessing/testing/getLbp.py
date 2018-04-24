from Code.Libraries import MyOculusLib as mol
from Code.Preprocessing import LBP, Gradients
from Code.Old import LBPtest as ltest

path = '../../../Images/all/'

mol.init(im_path=path, mouse_f=mol.draw_elipse)
lbp = LBP.LocalBinaryPatterns(40,3.5)

#mol.all_path(lbp.HistOnPath)
mol.all_path(lbp.LbpOnPath)






