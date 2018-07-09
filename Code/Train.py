#Tensorflow_gpu 1.8.0
#Keras 2.2
#keras preprocessing 1.0.1
#opencv_python 3.4.0.12
#Python 3.6 or 3.5.3
#keras applications 1.0.2
#scipy 1.0.1
#skimage (feature LBP - nieznana wersja)





from Code.Libraries import MyOculusLib as mol
from Code.Algorithms import Models as md
from Code.Libraries import DataAugmentationClasses as dac


#params
base_path='../'
image_size_level=20
base_scale=0.75
withMetricOrNo = 2
onlyWithMetric = False
onlyWithoutMetric = False
if(withMetricOrNo == 1):
	onlyWithMetric = True
if (withMetricOrNo == 2):
	onlyWithoutMetric = True
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

path = base_path+'Images/merged/'

class_mode='mask'

#model
show_function = mol.model_show_function
read_function = mol.read_and_size
validate_path_provider_func = mol.random_path
validate_start_path = base_path+'Images/all/'
filters=12

load_weights=True
weights_path="../weights/unet"
var_filename="../weights/var.txt"
validate=True

check_perf_times=0
check_perf_times_in_loop=0
loop_modulo = 20

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
	Mod.validate(onlyWithMetric=onlyWithMetric, onlyWithoutMetric=onlyWithoutMetric)
else:
	for i in range(total_ep):
		print("ep:"+str(i))
		Mod.save_weights()
		model.fit_generator(train_generator, steps_per_epoch=steps, epochs=ep)

		if i%loop_modulo==0:
			Mod.check_performance(train_generator, times=check_perf_times_in_loop)
