from keras import backend as K, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout


def create_model(input_shape, outputs):
    """
    Returns:
    model -- a Model() instance in Keras
    """

    cnn1 = Sequential()
    cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    cnn1.add(MaxPooling2D(pool_size=(2, 2)))
    cnn1.add(Dropout(0.2))

    cnn1.add(Flatten())

    cnn1.add(Dense(128, activation='relu'))
    cnn1.add(Dense(outputs, activation='softmax'))

    return cnn1
