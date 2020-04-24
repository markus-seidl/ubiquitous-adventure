from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense


def create_model(input_shape, outputs):
    """
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(32, (10, 10), strides=(10, 10), name='conv1')(X)
    X = BatchNormalization(axis=1, name='bn1')(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(5, name='dense_layer')(X)
    X = Dense(outputs, name='output_softmax', activation='softmax')(X)

    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='DummyModel')

    return model
