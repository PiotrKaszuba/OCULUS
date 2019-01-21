from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import LocalBinaryPatterns

path = '../../../Images/all/'

lbp = LocalBinaryPatterns.LocalBinaryPatterns(40, 3.5, method='default', equalize=False, negative=False, scale=0.2)

morn.all_path(lbp.LbpOnPath, path)
