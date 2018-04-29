import Code.Libraries.MyOculusLib as mol


#params
repo_base="../../../Images/"
repo_name="data"
new_name="merged"

old_repo = repo_base+repo_name+'/'
new_repo = repo_base + new_name+'/'


#go
mol.copyMaskData(old_repo,new_repo)
mol.createAllMasksForImagesCsv(new_repo)