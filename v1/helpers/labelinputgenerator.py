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

    def count_samples(self):
        return self.len

    def all_labels(self):
        """ This is len*batch_size long, and is longer than count_samples! """
        _, y = self.__data_generation(self.indexes, self.len)
        return y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.len / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes, self.batch_size)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.len)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, batch_size):
        """Generates data containing batch_size samples  # X : (n_samples, *dim, n_channels)"""
        # Initialization
        X = np.empty((batch_size, *self.dim))
        y = np.empty((batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(indexes):
            # Store sample
            img = self.data.read_idx(id)
            header, img = mx.recordio.unpack_img(img)

            temp = cv2.resize(img, dsize=(self.dim[0], self.dim[1]), interpolation=cv2.INTER_NEAREST)
            X[i,] = temp / 255  # convert to 0..1

            # Store class
            y[i] = header.label[4]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
