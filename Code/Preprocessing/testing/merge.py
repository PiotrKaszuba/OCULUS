from Code.Libraries import MyOculusLib as mol
from Code.Preprocessing.image_creation import MergeChannels as mh
from Code.Old import LBPtest as ltest

path = '../../../Images/all/'

mol.init(im_path=path, mouse_f=mol.draw_elipse)
merge = mh.MergeChannels(equalize=True, testMode=True)
mol.all_path(merge.MergeOnPath)