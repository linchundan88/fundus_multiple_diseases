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


def basic_block(filters, is_first_layer=False, block_seq=0):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        #block 0 an 1 are the same as resnet
        if block_seq == 0:  #after max pooling
            if is_first_layer:
                # don't repeat bn->relu since we just did maxpool
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(input)
            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(input)

            residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        elif block_seq == 1:
            if is_first_layer:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                      strides=(2, 2))(input)
            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(input)

            residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        # block 2 and block3 use dilated convolution
        elif block_seq == 2:
            if is_first_layer:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(1, 1))(input)
                residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(conv1)
            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(input)

                residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(conv1)

        elif block_seq == 3:
            if is_first_layer:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(input)
                residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(conv1)

            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(input)
                residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(conv1)

        return _shortcut(input, residual)

    return f


def bottleneck(filters, is_first_layer=False, block_seq=0):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):
        # block 0 an 1 are the same as resnet
        if block_seq == 0:
            if is_first_layer:
                # don't repeat bn->relu since we just did maxpool
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4))(input)
            else:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1))(input)
            conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
            residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

        elif block_seq == 1:
            if is_first_layer:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                        strides=(2, 2))(input)
            else:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1))(input)
            conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
            residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

        # block 2 and block3 use dilated convolution
        elif block_seq == 2:
            if is_first_layer:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                            dilation_rate=(1, 1))(input)
                conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(conv_1_1)
                residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(2, 2))(conv_3_3)
            else:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         dilation_rate=(2, 2))(input)
                conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(conv_1_1)
                residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(2, 2))(conv_3_3)

        elif block_seq == 3:
            if is_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         dilation_rate=(2, 2))(input)
                conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(conv_1_1)
                residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(4, 4))(conv_3_3)
            else:
                conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         dilation_rate=(4, 4))(input)
                conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(conv_1_1)
                residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(4, 4))(conv_3_3)

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


class DRN_A_Builder(object):
    @staticmethod
    def build(input_shape, num_classes, block_fn, repetitions, init_filters=64,
              init_kernal_size=7, regression=False, include_top=True):
        """Builds a custom ResNet like architecture.
        only use tf, (nb_channels, nb_rows, nb_cols) change to (nb_rows, nb_cols, nb_channels)
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_classes: The number of outputs at final softmax layer
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
        conv1 = _conv_bn_relu(filters=init_filters, kernel_size=(init_kernal_size, init_kernal_size),
                    strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1

        # filters = 64
        filters = init_filters
        # ResNet34 repetitions [3, 4, 6, 3]
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r,
                                    block_seq=i)(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        if include_top:
            x = GlobalAveragePooling2D()(block)

            # block_shape = K.int_shape(block)
            # pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
            #                          strides=(1, 1))(block)
            # flatten1 = Flatten()(pool2)

            if regression:
                dense = Dense(num_classes, activation='sigmoid')(x)
                # dense = Dense(units=num_classes)(flatten1)
            else:
                dense = Dense(units=num_classes, kernel_initializer="he_normal",
                              activation="softmax")(x)
                # dense = Dense(units=num_classes, kernel_initializer="he_normal",
                #               activation="softmax")(flatten1)

            model = Model(inputs=input, outputs=dense)
        else:
            model = Model(inputs=input, outputs=block)

        return model

    @staticmethod
    def build_DRN_A_18(input_shape, num_classes, include_top=True):
        # return DRN_A_Builder.build(input_shape, num_classes, basic_block, [2, 2, 2, 2],
        #                            init_filters=64, init_kernal_size=7, include_top=include_top)

        return DRN_A_Builder.build(input_shape, num_classes, basic_block, [2, 2, 2, 2],
                               init_filters=64, init_kernal_size=7, include_top=include_top)

    @staticmethod
    def build_DRN_A_34(input_shape, num_classes, include_top=True):
        return DRN_A_Builder.build(input_shape, num_classes, basic_block, [3, 4, 6, 3],
                                   init_filters=64, init_kernal_size=7, include_top=include_top)

    @staticmethod
    def build_DRN_A_50(input_shape, num_classes, include_top=True):
        return DRN_A_Builder.build(input_shape, num_classes, bottleneck, [3, 4, 6, 3],
                                   init_filters=64, init_kernal_size=7, include_top=include_top)

    @staticmethod
    def build_DRN_A_101(input_shape, num_classes, include_top=True):
        return DRN_A_Builder.build(input_shape, num_classes, bottleneck, [3, 4, 23, 3], include_top=include_top)

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    #params:21,307,650
    #original number of parameters, resnet34 :21.8M, resnet50:25.6M
    model1 = DRN_A_Builder.build_DRN_A_18(input_shape=(224, 224, 3), num_classes=2)

    model1.summary()

    print('OK')