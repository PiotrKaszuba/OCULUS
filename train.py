import DataAugmentationClasses as dac
import MyOculusLib as mol
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras import optimizers
from keras.optimizers import Adam
import numpy as np
import cv2
r = 7
load=False
def test(name):
	inp = mol.read_and_size(path="Images/validate/", name=name, scale=0.8)
	inp = np.reshape(inp, (1, 40, 40, 1))
	inp = inp / 255
	img = model.predict(inp)
	img = np.reshape(img, (40, 40))
	#ret, img =cv2.threshold(img, 0.66,1, cv2.THRESH_BINARY)
	mol.show(img)

model = Sequential()

model.add(Conv2D(20, (3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(40, 40,1)))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(20, (3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(20, (3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(20, (3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(1, (3,3), strides=(1,1), padding='same', activation="sigmoid"))



model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
if load:
	model.load_weights("weights")


for i in range(r):
	test(str(i))

f = dac.ImageDataGeneratorExtension(rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        zoom_range=0.05,
									rescale=1./255,
							        fill_mode='nearest')

train_generator = f.flow_from_directory_extension(directory="Images/trainTest/", color_mode="grayscale", class_mode='mask', target_size=(40,40))

model.fit_generator(train_generator, steps_per_epoch=30, epochs=50)

model.save_weights('weights')


for i in range(r):
	test(str(i))