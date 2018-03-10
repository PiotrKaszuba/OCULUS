import keras.preprocessing.image as i
from keras import backend as K
import numpy as np
import warnings
import os







class ImageDataGeneratorExtension(i.ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):

        super.__init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None)

        def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
            return NumpyArrayIteratorExtension(
                x, y, self,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                data_format=self.data_format,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format)

        def random_transform(self, x, y, seed=None):
            """Randomly augment a single image tensor.

            # Arguments
                x: 3D tensor, single image.
                seed: random seed.

            # Returns
                A randomly transformed version of the input (same shape).
            """
            # x is a single image, so it doesn't have image number at index 0
            img_row_axis = self.row_axis - 1
            img_col_axis = self.col_axis - 1
            img_channel_axis = self.channel_axis - 1

            if seed is not None:
                np.random.seed(seed)

            # use composition of homographies
            # to generate final transform that needs to be applied
            if self.rotation_range:
                theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            else:
                theta = 0

            if self.height_shift_range:
                tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
            else:
                tx = 0

            if self.width_shift_range:
                ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
            else:
                ty = 0

            if self.shear_range:
                shear = np.random.uniform(-self.shear_range, self.shear_range)
            else:
                shear = 0

            if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

            transform_matrix = None
            if theta != 0:
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
                transform_matrix = rotation_matrix

            if tx != 0 or ty != 0:
                shift_matrix = np.array([[1, 0, tx],
                                         [0, 1, ty],
                                         [0, 0, 1]])
                transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

            if shear != 0:
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                         [0, np.cos(shear), 0],
                                         [0, 0, 1]])
                transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

            if zx != 1 or zy != 1:
                zoom_matrix = np.array([[zx, 0, 0],
                                        [0, zy, 0],
                                        [0, 0, 1]])
                transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

            if transform_matrix is not None:
                hx, wx = x.shape[img_row_axis], x.shape[img_col_axis]
                hy, wy = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_x = i.transform_matrix_offset_center(transform_matrix, hx, wx)
                transform_matrix_y = i.transform_matrix_offset_center(transform_matrix, hy, wy)
                x = i.apply_transform(x, transform_matrix_x, img_channel_axis,
                                    fill_mode=self.fill_mode, cval=self.cval)

                y = i.apply_transform(y, transform_matrix_y, img_channel_axis,
                                      fill_mode=self.fill_mode, cval=self.cval)

            if self.channel_shift_range != 0:
                x = i.random_channel_shift(x,
                                         self.channel_shift_range,
                                         img_channel_axis)

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    x = i.flip_axis(x, img_col_axis)
                    y= i.flip_axis(y, img_col_axis)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    x = i.flip_axis(x, img_row_axis)
                    y = i.flip_axis(y, img_row_axis)
            return x, y


class NumpyArrayIteratorExtension(i.Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                          '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                          'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                          'However, it was passed an array with shape ' + str(self.x.shape) +
                          ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIteratorExtension, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_y = np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]
            x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            y = self.image_data_generator.standardize(y)
            batch_x[i] = x
            batch_y[i] = y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = i.array_to_img(batch_x[i], self.data_format, scale=True)
                img2 = i.array_to_img(batch_y[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                fname2 = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
                img2.save(os.path.join(self.save_to_dir+"masks/", fname))
        if self.y is None:
            return batch_x
        #batch_y = self.y[index_array]
        return batch_x, batch_y
