# Dilated Residual Networks https://arxiv.org/abs/1705.09914

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    GlobalAveragePooling2D
)

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


# Helper to build a BN -> relu block
def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

# Helper to build a conv -> BN -> relu block 只有 block之前的第一个conv使用 maxpooling之前的
def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    # 函数名可以作为函数的返回值
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

# Helper to build a BN -> relu -> conv block.  improved scheme
def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))

    # 函数名可以作为函数的返回值
    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, block_seq=0):
    """Builds a residual block with repeating blocks.
    每一个block的第一个conv层stride=2(除了第一个块之外，因为刚刚maxpooling)
    此外不需要_bn_relu_conv，直接conv
    """
    def f(input):
        for i in range(repetitions):
            input = block_function(filters=filters,
                    is_first_layer=(i == 0), block_seq=block_seq)(input)
        return input

    return f

def seperable_block(filters, is_first_layer=False, block_seq=0):

    def f(input):

        from keras import layers

        if block_seq == 0:

            residual = input

            x = layers.SeparableConv2D(filters, (3, 3),
                                       padding='same',
                                       use_bias=False)(input)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.SeparableConv2D(filters, (3, 3),
                                       padding='same',
                                       use_bias=False)(x)
            x = layers.BatchNormalization()(x)

            x = _shortcut(x, residual)


            x = layers.Activation('relu')(x)

        if block_seq in [1]:

            if is_first_layer:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           strides=(2, 2),
                                           use_bias=False)(input)
            else:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           use_bias=False)(input)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.SeparableConv2D(filters, (3, 3),
                                       padding='same',
                                       use_bias=False)(x)
            x = layers.BatchNormalization()(x)

            residual = x
            # x = _shortcut(x, residual)
            x = _shortcut(input, residual)

            x = layers.Activation('relu')(x)

        elif block_seq in [2]:

            if is_first_layer:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           dilation_rate=(1, 1),
                                           use_bias=False)(input)
            else:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           dilation_rate=(2, 2),
                                           use_bias=False)(input)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)


            x = layers.SeparableConv2D(filters, (3, 3),
                                       padding='same',
                                       dilation_rate=(2, 2),
                                       use_bias=False)(x)
            x = layers.BatchNormalization()(x)

            residual = x
            x = _shortcut(input, residual)

            x = layers.Activation('relu')(x)

        elif block_seq in [3]:

            if is_first_layer:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           dilation_rate=(2, 2),
                                           use_bias=False)(input)
            else:
                x = layers.SeparableConv2D(filters, (3, 3),
                                           padding='same',
                                           dilation_rate=(4, 4),
                                           use_bias=False)(input)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            x = layers.SeparableConv2D(filters, (3, 3),
                                       padding='same',
                                       dilation_rate=(4, 4),
                                       use_bias=False)(x)
            x = layers.BatchNormalization()(x)

            residual = x
            x = _shortcut(input, residual)


            x = layers.Activation('relu')(x)

        return x

    return f



def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class DRN_A_Xception_Builder(object):
    @staticmethod
    def build(input_shape, num_classes, block_fn, repetitions, init_filters=64,
              init_kernal_size=7, regression=False, include_top=True):

        _handle_dim_ordering()

        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=init_filters, kernel_size=(init_kernal_size, init_kernal_size),
                    strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1


        filters = [128, 256, 512, 1024]  # filters = [64, 128, 256, 512]
        # ResNet34 repetitions [3, 4, 6, 3]
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters[i], repetitions=r,
                                    block_seq=i)(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        if include_top:
            x = GlobalAveragePooling2D()(block)

            if regression:
                dense = Dense(num_classes, activation='sigmoid')(x)
            else:
                dense = Dense(units=num_classes, kernel_initializer="he_normal",
                              activation="softmax")(x)

            model = Model(inputs=input, outputs=dense)
        else:
            model = Model(inputs=input, outputs=block)

        return model


    @staticmethod
    def build_DRN_A_xception(input_shape, num_classes, include_top=True, regression=False):
        return DRN_A_Xception_Builder.build(input_shape, num_classes, seperable_block, [3, 4, 10, 3],
                                            include_top=include_top, regression=regression)


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #params:21,307,650
    #original number of parameters, resnet34 :21.8M, resnet50:25.6M
    model1 = DRN_A_Xception_Builder.build_DRN_A_xception(input_shape=(256, 256, 3), num_classes=2)

    model1.summary()

    print('OK')