import Models as md
import MyOculusImageLib as moil


def createOpticDiscModel(ModelName, gray=True, preprocessFunc=lambda x: x):
    FeatureName = "Tarcza"
    TrainModeName = FeatureName + ModelName
    image_size_level = 20
    base_scale = 0.75

    cols, rows = moil.getColsRows(level=image_size_level, base_scale=base_scale)

    if (gray):
        mode = 0
        channels_in = 1
        color_mode = 'grayscale'
    else:
        mode = 1
        channels_in = 3
        color_mode = 'rgb'
    filters = 12
    weights_path = "weights/unet" + TrainModeName
    var_filename = "weights/var" + TrainModeName + ".txt"
    Mod = md.Models(rows, cols, mode=mode, channels=channels_in,
                    weights_path=weights_path, var_filename=var_filename, read_func=moil.read_and_size,
                    preprocessFunc=preprocessFunc)
    Mod.get_model(filters=filters)
    Mod.load_weights()

    return Mod


def createAtophyModel(ModelName, gray=True, preprocessFunc=lambda x: x):
    FeatureName = "Zanik"
    TrainModeName = FeatureName + ModelName
    image_size_level = 10
    base_scale = 1.0

    cols, rows = moil.getColsRows(level=image_size_level, base_scale=base_scale)

    if (gray):
        mode = 0
        channels_in = 1
        color_mode = 'grayscale'
    else:
        mode = 1
        channels_in = 3
        color_mode = 'rgb'
    filters = 10
    weights_path = "weights/unet" + TrainModeName
    var_filename = "weights/var" + TrainModeName + ".txt"
    Mod = md.Models(rows, cols, mode=mode, channels=channels_in,
                    weights_path=weights_path, var_filename=var_filename, read_func=moil.read_and_size,
                    preprocessFunc=preprocessFunc)
    Mod.get_model(filters=filters)
    Mod.load_weights()

    return Mod


def createExitModel(ModelName, gray=True, preprocessFunc=lambda x: x):
    FeatureName = "Wyjscie"
    TrainModeName = FeatureName + ModelName
    image_size_level = 5
    base_scale = 1.0

    cols, rows = moil.getColsRows(level=image_size_level, base_scale=base_scale)

    if (gray):
        mode = 0
        channels_in = 1
        color_mode = 'grayscale'
    else:
        mode = 1
        channels_in = 3
        color_mode = 'rgb'
    filters = 8
    weights_path = "weights/unet" + TrainModeName
    var_filename = "weights/var" + TrainModeName + ".txt"
    Mod = md.Models(rows, cols, mode=mode, channels=channels_in,
                    weights_path=weights_path, var_filename=var_filename, read_func=moil.read_and_size,
                    preprocessFunc=preprocessFunc)
    Mod.get_model(filters=filters)
    Mod.load_weights()

    return Mod
