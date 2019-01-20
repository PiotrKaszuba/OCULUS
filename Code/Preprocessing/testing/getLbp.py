from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import LocalBinaryPatterns

path = '../../../Images/all/'

lbp = LocalBinaryPatterns.LocalBinaryPatterns(40, 3.5)

morn.all_path(lbp.LbpOnPath, path)
