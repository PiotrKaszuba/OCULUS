from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import MergeChannels as mh

path = '../../../Images/all/'

merge = mh.MergeChannels(equalize=True, testMode=True)
morn.all_path(merge.MergeOnPath, path)
