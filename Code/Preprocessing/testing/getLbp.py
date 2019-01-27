from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import LocalBinaryPatterns
from Code.Libraries import MyOculusImageLib as moil
path = '../../../Images/all/'

lbp = LocalBinaryPatterns.LocalBinaryPatterns(40, 3.5, method='default', equalize=True, negative=True, scale=1)

morn.all_path(lbp.LbpOnPath, path)
