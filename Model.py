import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def get_model(img_rows=512, img_cols=512, filters = 2):

    inputs = Input((img_rows, img_cols, 1))
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
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model
