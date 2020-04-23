import numpy as np
import keras
import mxnet as mx
import cv2

class LabelInputGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, base_filename, dim, n_classes, batch_size=32, shuffle=True):
        self.base_filename = base_filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.n_classes = n_classes

        self.data = mx.recordio.MXIndexedRecordIO(base_filename + '.idx', base_filename + '.rec', 'r')
        self.len = len(self.data.keys)

        # preparation
        self.indexes = np.arange(self.len)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.len / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.len)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples  # X : (n_samples, *dim, n_channels)"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(indexes):
            # Store sample
            img = self.data.read_idx(id)
            header, img = mx.recordio.unpack_img(img)
            # Copy the first self.dim pixels into the return value, ignore the rest (hard cropping + padding)
            # temp = np.zeros(self.dim)
            # d = (min(self.dim[0], img.shape[0]),
            #      min(self.dim[1], img.shape[1]),
            #      img.shape[2]
            #      )
            # temp[:, :, :] = img[0:d[0], 0:d[1], 0:d[2]]
            #X[i,] = keras.backend.resize_images(img, self.dim[0], self.dim[1], data_format="channels_last")

            # X[i, ] = mx.image.imresize(img, self.dim[0], self.dim[1])
            temp = cv2.resize(img, dsize=(self.dim[0], self.dim[1]), interpolation=cv2.INTER_NEAREST)

            X[i,] = temp/255 # convert to 0..1
            #X[i,] = temp
            # padd = [(0, self.dim[2] - img.shape[0]), (0, self.dim[1] - img.shape[1]), (0, 0)]
            # print(padd)
            # X[i,] = np.pad(img, padd, mode='constant', constant_values=0)
            # X[i, ] = mx.image.imresize(img, self.dim[0], self.dim[1])
            # X[i, ] = mx.ndarray.image.resize(data=img, size=(self.dim[0], self.dim[1]))
            # mxnet.ndarray.image.resize(data=None, size=_Null, keep_ratio=_Null, interp=_Null, out=None, name=None, **kwargs)

            # Store class
            y[i] = header.label[4]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
