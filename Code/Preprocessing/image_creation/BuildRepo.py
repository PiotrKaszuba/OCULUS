

#Builds a Repo based on existing repo (only images with mask) changing target image size and preprocessing images with function

import Code.Libraries.MyOculusLib as mol
import Code.Preprocessing.MergeChannels as mc
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