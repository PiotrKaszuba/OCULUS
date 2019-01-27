import os
import pickle
from operator import add
import cv2
import keras
from keras.layers import concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import *
from keras.optimizers import *
from matplotlib import pyplot as plt

import Code.Algorithms.Metrics as met
import Code.Libraries.MyOculusImageLib as moil
import Code.Libraries.MyOculusCsvLib as mocl

class Models:

    @staticmethod
    def model_show_function(x):
        y = []
        for i in range(len(x) - 1):
            y.append(x[i + 1])
        moil.show(x[0], other_im=y)

    def __init__(self, rowDim, colDim, mode=0, channels=1, out_channels=1, modify_col=False, row_div_col=0,
                 weights_path="../weights/unet", var_filename="../weights/var.txt", show_function=None, read_func=None,
                 validate_path_provider_func=None, validate_start_path=None, preprocessFunc = lambda x:x, constantVar = None):
        self.model = None
        self.constantVar = constantVar
        self.path = weights_path
        self.var_filename = var_filename
        self.rowDim = int(rowDim)
        self.colDim = int(colDim)
        self.mode = mode
        self.channels = channels
        self.out_channels = out_channels
        self.row_div_col = row_div_col
        self.show_function = show_function
        self.read_func = read_func
        self.validate_path_provider_func = validate_path_provider_func
        self.validate_start_path = validate_start_path
        self.preprocessFunc = preprocessFunc
        if row_div_col > 0:
            if modify_col:
                self.colDim = int(rowDim / row_div_col)
            else:
                self.rowDim = int(colDim * row_div_col)

    def save_weights(self):
        self.model.save_weights(self.path + '_' + str(self.var_file()))

    def load_weights(self):
        try:
            numb = self.var_file(True)
            if numb >= 0:
                self.model.load_weights(self.path + '_' + str(numb))
                return True
            return False
        except:
            print("Nie udało się wczytać wag sieci!")
            return False

    def centerDiffMetric(pred, x, y):
        return met.centerDiff(pred, x, y)

    def predict(self, im):
        w, h, c = moil.getWidthHeightChannels(im)

        if w != self.colDim or h!= self.rowDim or c!=self.channels:
            im = cv2.resize(im, (self.colDim, self.rowDim))[:, :]
        im = self.prepareImage(im)
        im = self.model.predict(im)
        im = moil.convertImageNetOutput(im)
        return im
    def prepareImage(self, im,  retboth = False):
        img = self.preprocessFunc(im)
        imgX = img.reshape((1, self.rowDim, self.colDim, self.channels))
        imgX = imgX / 255
        if retboth:
            return imgX, img
        return imgX
    def readImage(self, name, path, extension=".jpg"):
        return self.read_func(name=name, extension=extension, path=path, target_size=(self.colDim, self.rowDim), mode=0)
    def validate(self, pathForce=None, validateMode=0, preprocessFunc=lambda x: x, draw=True, onlyWithMetric=False,
                 onlyWithoutMetric=False, sumTimes=None, metrics=['distance', 'youden', 'jaccard', 'dice'], validTimes=1, weightsTimesValids=None):
        avgs, globals = (0, 0)
        for i in range(validTimes):
            if weightsTimesValids is not None:
                self.constantVar = i * weightsTimesValids
                self.load_weights()
            sum = [0]*len(metrics)
            confusion_matrix=[0]*4
            globalCount = False
            for metr in metrics:
                if 'global' in metr:
                    globalCount = True
            times = 0
            visited_path = {}
            while True:
                if pathForce is None:
                    path = self.validate_path_provider_func(self.validate_start_path, visited_path)
                    visited_path[path] = times

                else:
                    path = pathForce
                if path is None:
                    break
                if not os.path.exists(path):
                    continue
                images = os.listdir(path)
                for imp in images:  # len(os.listdir(path)) - 2):

                    true_path = path + 'mask/'
                    if not os.path.exists(os.path.join(path, imp)):
                        continue
                    if onlyWithMetric and not os.path.exists(os.path.join(true_path, imp)):
                        continue
                    else:
                        if onlyWithoutMetric and os.path.exists(os.path.join(true_path, imp)):
                            continue

                    im = self.read_func(name=imp, extension='', path=path, target_size=(self.colDim, self.rowDim), mode=0)
                    imgX, img = self.prepareImage(im,  retboth=True)
                    pred = self.model.predict(imgX)

                    pred = moil.convertImageNetOutput(pred)

                    toDraw = im if draw else None

                    x = [im, pred, img]
                    if os.path.exists(os.path.join(true_path, imp)):
                        true = self.read_func(name=imp, extension='', path=true_path, target_size=(self.colDim, self.rowDim))

                        true = true.reshape((self.rowDim, self.colDim, self.out_channels))
                        x.append(true)
                        results = met.customMetric(pred, true, toDraw=toDraw, metrics=metrics, globalCount=globalCount)
                        sum = list(map(add, sum, results[0]))

                        confusion_matrix = list(map(add, confusion_matrix, results[1]))

                        times += 1
                        if sumTimes is not None and times >= sumTimes:
                            break
                    else:
                        met.draw(pred, toDraw)

                    if sumTimes is None:
                        self.show_function(x)

            avgs = [x / times for x in sum]
            strgSum = ''
            strgAvgs = ''
            for val in sum:
                strgSum += str(val) + ', '
            for val in avgs:
                strgAvgs += str(val) + ', '

            globals = []
            if globalCount:
                globals = met.globals(confusion_matrix)
                print("Global Jaccard: " + str(globals[0]) + ", Global Dice: " + str(globals[1]))
            print("Times: " + str(times) + ", sums: " + strgSum + "Average metrics: " + strgAvgs)
            self.validate_to_csv(metrics, avgs+globals)
        return avgs+globals

    def validate_to_csv(self, metrics, values):
        path = self.path.split('/')[:-1]
        path='/'.join(path)
        epoch = (self.var_file(True)%100)*100
        epoch = epoch + int(self.var_file(True)/100)*100
        mocl.writeToCsv(path+'/scores.csv', ['name']+metrics, [self.path.split('/')[-1]+'_'+str(epoch)]+values)
    def check_performance(self, validate_generator, times=1, metrics=['distance', 'youden', 'jaccard', 'dice']):
        for i in range(times):
            pic = validate_generator.next()

            true = pic[1][0]

            pred = self.model.predict(pic[0][0].reshape(1, self.rowDim, self.colDim, self.channels))
            pred = moil.convertImageNetOutput(pred)
            true = moil.convertImageNetOutput(true)

            met.customMetric(pred, true, metrics=metrics)
            x = []

            x.append(pic[0][0].reshape((self.rowDim, self.colDim, self.channels)))
            x.append(true)
            x.append(pred)

            if self.show_function != None:
                self.show_function(x)

    def var_file(self, read=False, increase = 1):
        if self.constantVar is not None:
            return self.constantVar
        numb = -1
        if not os.path.isfile(self.var_filename):
            fo = open(self.var_filename, "w")
            fo.write("-1")
            fo.close()
        else:
            fo = open(self.var_filename, "r")
            numb = int(next(fo))
            fo.close()

        if read:
            return numb
        if numb < 0:
            numb =0
        else:
            numb += increase
        fo = open(self.var_filename, "w")
        fo.write(str(numb))
        fo.close()
        return numb

    def load_loss(self, epoch):
        try:
            with open(self.path + "_losses_" + str(epoch), 'rb') as handle:
                return pickle.load(handle)
        except:
            return None

    def plot_loss(self, epoch=None):
        b = None
        if epoch is None:
            epoch = self.var_file(read=True)
        b = self.load_loss(epoch)
        if b is None:
            return
        plt.plot(list(range(epoch+1-len(b), epoch + 1)), b, label="loss")
        plt.legend()
        plt.show()

    def get_model(self, filters=2, le=1e-04, decay=0):

        if self.model != None:
            return self.model

        inputs = Input((self.rowDim, self.colDim, self.channels))
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        model.compile(optimizer=Adam(lr=le, decay=decay), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

        return self.model


class Callbacks(keras.callbacks.Callback):

    def __init__(self, ModelClass, save_modulo_epochs=None, collectLoss=False, printDecay=False, message=None):
        self.ModelClass = ModelClass
        self.save_modulo_epochs = save_modulo_epochs
        self.printDecay = printDecay
        self.collectLoss = collectLoss
        try:
            epoch = ModelClass.var_file(read=True)
            losses = ModelClass.load_loss(epoch)
        except:
            losses = []
        if losses is None:
            losses = []
        self.losses = losses
        self.message = message

    def on_epoch_end(self, epoch, logs={}):
        if self.message is not None:
            print(self.message)
        if self.printDecay:
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            print(K.eval(iterations))
            print(K.eval(lr_with_decay))

        self.losses.append(logs.get('loss'))
        if self.save_modulo_epochs is not None:
            if (epoch + 1) % self.save_modulo_epochs == 0:
                self.ModelClass.save_weights()
                if self.collectLoss:
                    with open(self.ModelClass.path + '_losses_' + str(len(self.losses)), 'wb') as handle:
                        pickle.dump(self.losses, handle)

        return
