# Tensorflow_gpu 1.8.0
# Keras 2.2
# keras preprocessing 1.0.1
# opencv_python 3.4.0.12
# Python 3.6 or 3.5.3
# keras applications 1.0.2
# scipy 1.0.1
# skimage (feature LBP - nieznana wersja)


from Code.Algorithms import Models as md
from Code.Libraries import MyOculusImageLib as moil
from Code.Libraries import MyOculusRepoNav as morn
from Code.Preprocessing import DataAugmentationClasses as dac
from operator import add
# params
base_path = '../'
image_size_level = 10
base_scale = 1.0
withMetricOrNo = 1
onlyWithMetric = False
onlyWithoutMetric = False
if withMetricOrNo == 1:
    onlyWithMetric = True
if withMetricOrNo == 2:
    onlyWithoutMetric = True
batch_size = 32
total_ep = 1
ep = 500
steps = 10

cols, rows = moil.getColsRows(level=image_size_level, base_scale=base_scale)
gray = True

if (gray):
    mode = 0
    channels_in = 1
    color_mode = 'grayscale'
else:
    mode = 1
    channels_in = 3
    color_mode = 'rgb'

# data augmentation
aug = dac.getAugmentationParamsZanik()

FeatureName = "Zanik"
TrainModeName = FeatureName + "Gray50Cross5"

class_mode = 'mask'
#

# model
show_function = md.Models.model_show_function
read_function = moil.read_and_size
validate_path_provider_func = morn.next_path

filters = 10

load_weights = True
save_modulo = 100

validate = True
metrics = ["youden", "global_youden", "global_jaccard", 'global_dice']
# mer = mc.MergeChannels(True)
validatePreprocessFunc = lambda x: x
draw = False
sumTimes = 10

check_perf_times = 0
check_perf_times_in_loop = 0
loop_modulo = 1

learn_rate = 3e-04
decay_rate = 5e-04
printDecay = True

collectLoss = True
# setup
f = dac.ImageDataGeneratorExtension(rotation_range=aug['rotation_range'],
                                    width_shift_range=aug['width_shift_range'],
                                    height_shift_range=aug['height_shift_range'],
                                    zoom_range=aug['zoom_range'],
                                    shear_range=aug['shear_range'],
                                    rescale=aug['rescale'],
                                    fill_mode=aug['fill_mode'])


CrossTimes = 5

sum = [0]*len(metrics)
for j in range(CrossTimes):




    path = base_path + 'Images/' + FeatureName +'_' + str(j+1) +'_train' + '/'
    validate_start_path = base_path + 'Images/' + FeatureName +'_' + str(j+1) +'_test' + '/'
    weights_path = "../weights/unet" + TrainModeName + '_' + str(j+1)
    var_filename = "../weights/var" + TrainModeName + '_' + str(j+1) + ".txt"
    train_generator = f.flow_from_directory_extension(directory=path, batch_size=batch_size, color_mode=color_mode,
                                                  class_mode=class_mode, target_size=(rows, cols))
    Mod = md.Models(rows, cols, mode=mode, channels=channels_in, show_function=show_function, read_func=read_function,
                validate_path_provider_func=validate_path_provider_func, validate_start_path=validate_start_path,
                weights_path=weights_path, var_filename=var_filename)

    # model creation
    model = Mod.get_model(filters=filters, le=learn_rate, decay=decay_rate)
    weights_loaded = False
    if load_weights:
        weights_loaded = Mod.load_weights()
    if not weights_loaded:
        Mod.save_weights()


    Mod.check_performance(train_generator, times=check_perf_times, metrics= metrics)

    callbacks = md.Callbacks(ModelClass=Mod, save_modulo_epochs=save_modulo, printDecay=printDecay, collectLoss=collectLoss, message='Model no.: '+str(j+1))
    # go
    if validate:
        results = Mod.validate(validateMode=mode, preprocessFunc=validatePreprocessFunc, draw=draw, onlyWithMetric=onlyWithMetric,
                     onlyWithoutMetric=onlyWithoutMetric, sumTimes=sumTimes, metrics=metrics, validTimes=6, validName='Zanik', weightsTimesValids=100)
        sum = list(map(add, sum, results))
    else:
        for loop in range(total_ep):
            i = loop + 1
            print("ep:" + str(i))

            model.fit_generator(train_generator, steps_per_epoch=steps, epochs=ep, callbacks=[callbacks])

            if i % loop_modulo == 0:
                Mod.check_performance(train_generator, times=check_perf_times_in_loop)


avgs = [x / CrossTimes for x in sum]
print(avgs)