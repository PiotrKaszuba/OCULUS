from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import Gradients

path = '../../../Images/all/'

grad = Gradients.Gradients(equalize=True, scale=0.3)

morn.all_path(grad.GradientSumOnPath, path)
