#coding=utf-8
#https://github.com/raghakot/keras-resnet
from __future__ import division

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
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

# http://arxiv.org/pdf/1603.05027v2.pdf
# 只是增加了 init_filters（原来固定64）和init_kernal_size（原来固定7）的可定义
# 以及build_resnet_mymodel_34_32_3 定义了一些models, 针对大模型 增加一个 blocck

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

    # 函数名可以作为函数的返回值
    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
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


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating blocks.
    每一个block的第一个conv层stride=2(除了第一个块之外，因为刚刚maxpooling)
    此外不需要_bn_relu_conv，直接conv
    """
    def f(input):
        for i in range(repetitions):
            if i == 0 and not is_first_layer: # is_first_layer just after maxpooling
                init_strides = (2, 2)
            else:
                init_strides = (1, 1)

            #第一层第一个conv padding 和 kernel_initializer 与其他不同
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

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


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, init_filters=64,
              init_kernal_size=7, regression=False):
        """Builds a custom ResNet like architecture.
        only use tf, (nb_channels, nb_rows, nb_cols) change to (nb_rows, nb_cols, nb_channels)
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        # if K.image_dim_ordering() == 'tf':          #tf or th   new version image_data_format: "channels_last",
        #     input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        # conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        conv1 = _conv_bn_relu(filters=init_filters, kernel_size=(init_kernal_size, init_kernal_size),
                    strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1

        # filters = 64
        filters = init_filters
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)

        if regression:
            dense = Dense(units=num_outputs)(flatten1)
        else:
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                          activation="softmax")(flatten1)

        # gap = GlobalAveragePooling2D()(block)
        # if regression:
        #     dense = Dense(units=num_outputs)(gap)
        # else:
        #     dense = Dense(units=num_outputs, kernel_initializer="he_normal",
        #                   activation="softmax")(gap)

        model = Model(inputs=input, outputs=dense)

        return model

    @staticmethod
    def build_custom_filters(input_shape, num_outputs, block_fn, repetitions, filters,
                             init_kernal_size=5, regression=False):

        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)

        conv1 = _conv_bn_relu(filters=filters[0], kernel_size=(init_kernal_size, init_kernal_size),
                    strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1

        for i, r in enumerate(repetitions):
            current_filters = filters[i]
            block = _residual_block(block_fn, filters=current_filters, repetitions=r, is_first_layer=(i == 0))(block)

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)

        if regression:
            dense = Dense(units=num_outputs)(flatten1)
        else:
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                          activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)

        return model

    #region original predefined ResNet different number of layers

    #region Regresion
    @staticmethod
    def build_resnet_regression_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3]
                                   ,regression=True)

    @staticmethod
    def build_resnet_regression_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3]
                                   ,regression=True)

    #endregion
    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
    #endregion

    #region 青光眼，视神经萎缩，检测视盘内 (112*112) 减少一个residual block
    @staticmethod
    def build_resnet_optic_disk1(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 8, 3],
                    init_filters=64, init_kernal_size=5)

    def build_resnet_optic_disk2(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 3],
                    init_filters=64, init_kernal_size=5)

    @staticmethod
    def build_resnet_optic_disk_112(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 8, 3],
                    init_filters=64, init_kernal_size=5)

    # endregion

    #region  my design  ResNet 448
    @staticmethod
    # original resnet filters [64,128,256,512]
    def build_resnet_448_1(input_shape, num_outputs):
        return ResnetBuilder.build_custom_filters(input_shape, num_outputs, bottleneck,
                repetitions=[2, 3, 4, 5, 3], filters=[64, 128, 256, 384, 512], init_kernal_size=5)

    @staticmethod
    # Total params: 80,479,108
    # Trainable params: 80,450,308
    # Non-trainable params: 28,800
    def build_resnet_448_34_64_5(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 3, 4, 5, 3],
                    init_filters=64, init_kernal_size=5)

    @staticmethod
    def build_resnet_448_51_64_5(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [2, 3, 4, 5, 3],
                    init_filters=64, init_kernal_size=5)


    # endregion

if __name__ == '_main_':
    model = ResnetBuilder.build_resnet_448_1((448, 448, 3), 2)
    model.summary()