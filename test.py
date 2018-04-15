import MyOculusLib as mol
import Model as md
import DataAugmentationClasses as dac
import os

def check_performance(model, validate_generator):
	pic = validate_generator.next()

	pred = model.predict(pic[0][0].reshape(1,rows,cols,1))
	x = []
	for i in range(len(pic)):
		x.append(pic[i][0].reshape((rows, cols)))

	x.append(pred.reshape((rows, cols)))
	mol.show(x[0], other_im=[x[1], x[2]])


def var_file(read=False):
	numb = 0
	if not os.path.isfile("weights/var.txt"):
		fo = open("weights/var.txt", "w")
		fo.write("0")
		fo.close()
	else:
		fo = open("weights/var.txt", "r")
		numb = int(next(fo))
		fo.close()

	if read:
		return numb
	numb += 1
	fo = open("weights/var.txt", "w")
	fo.write(str(numb))
	fo.close()
	return numb


mol.init()
scale=0.8
cols=240
rows=int(cols*scale)

f = dac.ImageDataGeneratorExtension(rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        zoom_range=0.05,
									rescale=1./255,
							        fill_mode='nearest')

train_generator = f.flow_from_directory_extension(directory="Images//", batch_size=32, color_mode="grayscale", class_mode='mask', target_size=(rows,cols))
model = md.get_model(rows, cols)
model.load_weights("weights/unet"+str(var_file(True)))
check_performance(model, train_generator)

for i in range(1000):
	model.fit_generator(train_generator, steps_per_epoch=10, epochs=1)

	model.save_weights('weights/unet' + str(var_file()))

	#check_performance(model, train_generator)
#mol.all_path(lambda x : mol.show(mol.read_and_size("0", scale=0, h_div_w=0.8, not_modified_wanted_value=400,  path = x)) )