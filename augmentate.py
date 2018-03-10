import Unet.data as dt
import MyOculusLib as mol

mol.init()
img = mol.read_and_size("0", mol.random_path())
x = dt.myAugmentation()
img = img.reshape(img.shape +(1,1,))
x.doAugmentate(img, save_to_dir="Images/augmentate",save_prefix="aug",save_format="jpg", batch_size=20)



