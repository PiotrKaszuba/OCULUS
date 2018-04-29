from Code.Libraries import MyOculusLib as mol
from Code.Algorithms import Models as md
from Code.Libraries import DataAugmentationClasses as dac


#params
base_path='../'
image_size_level=15
base_scale=0.75

batch_size = 32
total_ep = 1000
ep = 1
steps = 10

cols, rows = md.Models.getColsRows(level=image_size_level, base_scale=base_scale)
gray=False

if(gray):
	mode=0
	channels_in =1
	color_mode='grayscale'
else:
	mode=1
	channels_in=3
	color_mode='rgb'


#data augmentation
aug= dac.getAugmentationParams()

path = base_path+'Images/data/'

class_mode='mask'

#model
show_function = mol.model_show_function
read_function = mol.read_and_size
validate_path_provider_func = mol.random_path
validate_start_path = base_path+'Images/awaiting/'

filters=1

load_weights=True
weights_path="../weights/unet"
var_filename="../weights/var.txt"
validate=False
check_perf_times=3
check_perf_times_in_loop=0
learn_rate = 1e-04
#setup
mol.init(im_path=path)
f = dac.ImageDataGeneratorExtension(rotation_range=aug['rotation_range'],
							        width_shift_range=aug['width_shift_range'],
							        height_shift_range=aug['height_shift_range'],
							        zoom_range=aug['zoom_range'],
									shear_range=aug['shear_range'],
									rescale=aug['rescale'],
							        fill_mode=aug['fill_mode'])
train_generator = f.flow_from_directory_extension(directory=path, batch_size=batch_size, color_mode=color_mode, class_mode=class_mode, target_size=(rows,cols))
Mod = md.Models(rows, cols, mode=mode, channels=channels_in, show_function=show_function, read_func=read_function, validate_path_provider_func=validate_path_provider_func, validate_start_path=validate_start_path, weights_path=weights_path, var_filename=var_filename)

#model creation
model = Mod.get_model(filters=filters, le=learn_rate)
if load_weights:
	Mod.load_weights()

Mod.check_performance(train_generator, times=check_perf_times)

#go
if validate:
	Mod.validate()
else:
	for i in range(total_ep):
		print("ep:"+str(i))
		Mod.save_weights()
		model.fit_generator(train_generator, steps_per_epoch=steps, epochs=ep)

		Mod.check_performance(train_generator, times=check_perf_times_in_loop)
