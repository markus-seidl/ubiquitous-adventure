from keras import backend as K, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, ReLU, DepthwiseConv2D, Dense, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


def create_model(input_shape, outputs):
    """
    Returns:
    model -- a Model() instance in Keras
    """

    alpha = 0.5
    depth_multiplier = 1
    dropout = 0.2

    X_input = Input(input_shape)
    x = _conv_block(X_input, 32, alpha, strides=(2, 2))
    #x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    #x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
    #                          strides=(2, 2), block_id=2)
    #x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Dense(outputs, activation='sigmoid')

    return Model(inputs=X_input, outputs=x, name='simplemobilenet')


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=-1, name='conv1_bn')(x)  # channels_last = -1
    return ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = -1  ### 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)),
                          name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                        padding='same' if strides == (1, 1) else 'valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
