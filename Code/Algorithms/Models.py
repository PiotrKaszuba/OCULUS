import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import Code.Algorithms.Metrics as met
class Models:
    row_multi = 16
    col_multi = 16

    @staticmethod
    def getColsRows(level, base_scale, use_col=True):
        if use_col:
            cols = int(Models.col_multi * level)
            rows = int(round(level*base_scale)*Models.row_multi)
        else:
            rows = int(Models.row_multi * level)
            cols = int(round(level * base_scale) * Models.col_multi)

        return cols,rows

    def __init__(self, rowDim, colDim, mode=0, channels=1, out_channels=1, modify_col= False, row_div_col=0, weights_path="../weights/unet", var_filename="../weights/var.txt", show_function=None, read_func=None, validate_path_provider_func= None, validate_start_path=None):
        self.model = None
        self.path= weights_path
        self.var_filename= var_filename
        self.rowDim = int(rowDim)
        self.colDim = int(colDim)
        self.mode=mode
        self.channels = channels
        self.out_channels=out_channels
        self.row_div_col = row_div_col
        self.show_function = show_function
        self.read_func = read_func
        self.validate_path_provider_func = validate_path_provider_func
        self.validate_start_path = validate_start_path

        if row_div_col > 0:
            if modify_col:
                self.colDim = int(rowDim/row_div_col)
            else:
                self.rowDim = int(colDim*row_div_col)

    def save_weights(self):
        self.model.save_weights(self.path + str(self.var_file()))

    def load_weights(self):
        numb = self.var_file(True)
        if numb > 0:
            self.model.load_weights(self.path + str(numb))

    def metr(pred, true):
        return met.customMetric(pred, true)

    def validate(self, pathForce=None):

        while True:
            if pathForce == None:
                path = self.validate_path_provider_func(start_path=self.validate_start_path)
            else:
                path = pathForce
            for i in range(len(os.listdir(path)) - 2):
                img = self.read_func(name=str(i), path=path, target_size=(self.colDim, self.rowDim), mode=self.mode)
                imgX = img.reshape((1, self.rowDim, self.colDim, self.channels))
                imgX = imgX / 255
                pred = self.model.predict(imgX)
                pred = pred.reshape((self.rowDim, self.colDim))

                x = [img, pred]
                self.show_function(x)

    def check_performance(self, validate_generator, times=1):
        for i in range(times):
            pic = validate_generator.next()

            true = pic[1][0].reshape((self.rowDim, self.colDim, self.out_channels))

            pred = self.model.predict(pic[0][0].reshape(1,self.rowDim,self.colDim,self.channels))
            pred = pred.reshape((self.rowDim, self.colDim))
            #print(met.centerDiff(pred,true,check=False))
            #print(met.binaryDiff(pred,true))
            print("Custom metric: " +str(met.customMetric(pred, true)))
            x = []

            x.append(pic[0][0].reshape((self.rowDim, self.colDim,self.channels)))
            x.append(true)
            x.append(pred)
            if self.show_function != None:
                self.show_function(x)



    def var_file(self, read=False):
        numb = 0
        if not os.path.isfile(self.var_filename):
            fo = open(self.var_filename, "w")
            fo.write("0")
            fo.close()
        else:
            fo = open(self.var_filename, "r")
            numb = int(next(fo))
            fo.close()

        if read:
            return numb
        numb += 1
        fo = open(self.var_filename, "w")
        fo.write(str(numb))
        fo.close()
        return numb

    def get_model(self, filters = 2, le=1e-04):

        if self.model != None:
            return self.model

        inputs = Input((self.rowDim, self.colDim, self.channels))
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)


        model = Model(input=inputs, output=conv10)
        model.compile(optimizer=Adam(lr=le), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

        return self.model
