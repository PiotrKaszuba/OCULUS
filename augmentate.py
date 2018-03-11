import DataAugmentationClasses as dac
import MyOculusLib as mol

mol.init()
x = dac.ImageDataGeneratorExtension(rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

i = 0
#directory = mol.random_path()
#mol.mask_on_path(directory)
for batch in x.flow_from_directory_extension(directory="Images/all/", color_mode="grayscale", save_to_dir="Images/augmentate/", class_mode='mask'):
    i+=1
    if i>=3:
        break



