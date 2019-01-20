import os

from keras.layers import concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import *
from keras.optimizers import *
from operator import add, truediv
import Code.Algorithms.Metrics as met
import Code.Libraries.MyOculusImageLib as moil


class Models:

    @staticmethod
    def model_show_function(x):
        y = []
        for i in range(len(x) - 1):
            y.append(x[i + 1])
        moil.show(x[0], other_im=y)

    def __init__(self, rowDim, colDim, mode=0, channels=1, out_channels=1, modify_col=False, row_div_col=0,
                 weights_path="../weights/unet", var_filename="../weights/var.txt", show_function=None, read_func=None,
                 validate_path_provider_func=None, validate_start_path=None):
        self.model = None
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
            if numb > 0:
                self.model.load_weights(self.path + '_' + str(numb))
        except:
            print("Nie udało się wczytać wag sieci!")

    def centerDiffMetric(pred, x, y):
        return met.centerDiff(pred, x, y)

    def validate(self, pathForce=None, validateMode=0, preprocessFunc=lambda x: x, draw=True, onlyWithMetric=False,
                 onlyWithoutMetric=False, sumTimes = None):
        sum = [0, 0]
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
            for i in range(50):  # len(os.listdir(path)) - 2):

                true_path = path + 'mask/'
                if not os.path.exists(os.path.join(path, str(i) + '.jpg')):
                    continue
                if onlyWithMetric and not os.path.exists(os.path.join(true_path, str(i) + '.jpg')):
                    continue
                else:
                    if onlyWithoutMetric and os.path.exists(os.path.join(true_path, str(i) + '.jpg')):
                        continue

                im = self.read_func(name=str(i), path=path, target_size=(self.colDim, self.rowDim), mode=validateMode)
                img = preprocessFunc(im)
                imgX = img.reshape((1, self.rowDim, self.colDim, self.channels))
                imgX = imgX / 255
                pred = self.model.predict(imgX)
                pred = pred.reshape((self.rowDim, self.colDim))
                toDraw = im if draw else None
                if os.path.exists(os.path.join(true_path, str(i) + '.jpg')):
                    true = self.read_func(name=str(i), path=true_path, target_size=(self.colDim, self.rowDim))

                    results = met.customMetric(pred, true, check=False, toDraw=toDraw)
                    sum = list(map(add, sum, results))
                    times+=1
                    if sumTimes is not None and times >= sumTimes:
                        break
                else:
                    met.draw(pred, toDraw)
                x = [im, pred, img]
                if sumTimes is None:
                    self.show_function(x)

        avgs = [x / times for x in sum]
        print("Times: " + str(times) +", sums: " + str(sum[0]) + ", " + str(sum[1]) + ", Average metrics: " + str(avgs[0]) + ", " + str(avgs[1]))

    def check_performance(self, validate_generator, times=1):
        for i in range(times):
            pic = validate_generator.next()

            true = pic[1][0].reshape((self.rowDim, self.colDim, self.out_channels))

            pred = self.model.predict(pic[0][0].reshape(1, self.rowDim, self.colDim, self.channels))
            pred = pred.reshape((self.rowDim, self.colDim))
            # print(met.centerDiff(pred,true,check=False))
            # print(met.binaryDiff(pred,true))
            print("Custom metric: " + str(met.customMetric(pred, true)))
            x = []

            x.append(pic[0][0].reshape((self.rowDim, self.colDim, self.channels)))
            x.append(true)
            x.append(pred)
            '''
            import cv2
            ims = pred*255
            unique, counts = np.unique(pred, return_counts=True)
            norm_image = cv2.normalize(ims, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            #norm_image = norm_image.astype(np.uint8)

            equ = cv2.equalizeHist(norm_image)

            x.append(equ)
            '''
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

    def get_model(self, filters=2, le=1e-04):

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
        model.compile(optimizer=Adam(lr=le), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

        return self.model
