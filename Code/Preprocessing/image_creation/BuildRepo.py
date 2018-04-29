import Code.Libraries.MyOculusLib as mol
import numpy as np
import Code.Preprocessing.image_creation.MergeChannels as mc
import Code.Algorithms.Models as md

#init
equalize = True
mer= mc.MergeChannels(equalize)


#params
repo_base="../../../Images/"
repo_name="data"
new_name="merged"

old_repo = repo_base+repo_name+'/'
new_repo = repo_base + new_name+'/'
level = 15
scale = 0.75
target_size= md.Models.getColsRows(level, scale)
onlyMasked = True
override = True
#go
mol.createFromAllPathImageAfterFunction(old_repo,new_repo, mer.Merge, target_size=target_size, onlyMasked=onlyMasked, override=override)