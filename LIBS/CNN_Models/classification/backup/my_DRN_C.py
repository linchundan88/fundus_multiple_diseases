# Dilated Residual Networks https://arxiv.org/abs/1705.09914

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
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

    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1) )

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


def _residual_block(block_function, filters, repetitions, is_first_block=False, block_seq=0):
    def f(input):
        for i in range(repetitions):
            if i == 0 and (block_seq in [1, 2, 3]): #2,3,4,start 0, 第一个block不需要stride=2
                init_strides = (2, 2)
            else:
                init_strides = (1, 1)

            #第一层第一个conv padding 和 kernel_initializer 与其他不同
            input = block_function(filters=filters, init_strides=init_strides,
                   is_first_block_of_first_layer=(is_first_block and i == 0), block_seq=block_seq)(input)

        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, block_seq=0):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if block_seq == 4:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=(1, 1), dilation_rate=(2, 2))(input)

            conv2 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=(1, 1), dilation_rate=(2, 2))(conv1)

        elif block_seq == 5:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=(1, 1), dilation_rate=(4, 4))(input)

            conv2 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=(1, 1), dilation_rate=(4, 4))(conv1)
        else:
            if is_first_block_of_first_layer:  #DRN-C 前面是第一个_conv_bn_relu
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(input)
            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)

            conv2 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=(1, 1))(conv1)

        return _shortcut(input, conv2)

    return f

#DRN-C the last two blocks
def basic_block_without_residual(filters, dilation_rate=(2, 2)):
    def f(input):
        conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                              dilation_rate=dilation_rate)(input)

        conv2 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                              dilation_rate=dilation_rate)(conv1)

        return conv2

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, block_seq=0):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if block_seq == 4:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         strides=1, dilation_rate=(2, 2))(input)

            conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2))(conv_1_1)
            residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(2, 2))(conv_3_3)

        elif block_seq == 5:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         strides=1, dilation_rate=(4, 4))(input)

            conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=(4, 4))(conv_1_1)
            residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1), dilation_rate=(4, 4))(conv_3_3)
        else:
            if is_first_block_of_first_layer:   #DRN-C  前面是第一个_conv_bn_relu
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=(1, 1),
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

#DRN-C the last two blocks
def bottleneck_without_residual(filters,dilation_rate=(2, 2)):
    def f(input):
        conv1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                              strides=(1, 1), dilation_rate=dilation_rate)(input)

        conv2 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                              dilation_rate=dilation_rate)(conv1)

        conv3 = _bn_relu_conv(filters=filters*4, kernel_size=(1, 1),
                              dilation_rate=dilation_rate)(conv2)

        return conv3

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


class DRN_C_Builder(object):
    @staticmethod
    def build_DRN_C(input_shape, num_classes=2, include_top=True,
            block_fn_str='basic_block', repetitions=[1, 1, 2, 2, 2, 2, 1, 1],
            filters=(16, 32, 64, 128, 256, 512, 512, 512),
            regression=False):

        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        block_fn = _get_block(block_fn_str)

        input = Input(shape=input_shape)

        conv1 = _conv_bn_relu(filters=filters[0], kernel_size=(7, 7),
                              strides=(1, 1))(input)
        block = conv1

        repetitions1 = repetitions[:-2] #最后两个除外
        for i, r in enumerate(repetitions1):
            # DRN-C do not use max pooling, add level 1,2 two extra top levels

            block = _residual_block(block_fn, filters=filters[i], repetitions=r,
                                    is_first_block=(i == 0), block_seq=i)(block)

        # the last two blocks
        if block_fn_str == 'basic_block':
            for _ in range(repetitions[6]):
                block = basic_block_without_residual(filters[6], dilation_rate=(2, 2))(block)

            for _ in range(repetitions[7]):
                block = basic_block_without_residual(filters[7], dilation_rate=(1, 1))(block)
        elif block_fn_str == 'bottleneck':
            for _ in range(repetitions[6]):
                block = bottleneck_without_residual(filters[6], dilation_rate=(2, 2))(block)

            for _ in range(repetitions[7]):
                block = bottleneck_without_residual(filters[7], dilation_rate=(1, 1))(block)

        # Last activation
        block = _bn_relu(block)

        if include_top:
            # Classifier block
            block_shape = K.int_shape(block)

            pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                     strides=(1, 1))(block)
            flatten1 = Flatten()(pool2)
            if regression:
                dense = Dense(units=num_classes)(flatten1)
            else:
                dense = Dense(units=num_classes, kernel_initializer="he_normal",
                              activation="softmax")(flatten1)

            model = Model(inputs=input, outputs=dense)
        else:
            model = Model(inputs=input, outputs=block)

        return model

    @staticmethod
    def build_DRN_C_26(input_shape, num_classes=2, include_top=True):
        return DRN_C_Builder.build_DRN_C(input_shape=input_shape,
                                         block_fn_str='basic_block',
                                         include_top=include_top,
                                         num_classes=num_classes, regression=False,
                                         repetitions=[1, 1, 2, 2, 2, 2, 1, 1])
    @staticmethod
    def build_DRN_C_42(input_shape, num_classes=2, include_top=True):
        return DRN_C_Builder.build_DRN_C(input_shape=input_shape,
                                         block_fn_str='basic_block',
                                         include_top=include_top,
                                         num_classes=num_classes, regression=False,
                                         repetitions=[1, 1, 3, 4, 6, 3, 1, 1])
    @staticmethod
    def build_DRN_C_58(input_shape, num_classes=2, include_top=True):
        return DRN_C_Builder.build_DRN_C(input_shape=input_shape,
                                         block_fn_str='bottleneck',
                                         include_top=include_top,
                                         num_classes=num_classes, regression=False,
                                         repetitions=[1, 1, 3, 4, 6, 3, 1, 1])

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # inception-resnet-v2 parameters:54,339,810

    #params:20,631,682 最后几层conv:2,359,808
    # model1 = DRN_C_Builder.build_DRN_C_26(input_shape=(224, 224, 3), num_classes=2)

    #params:30,750,978
    model1 = DRN_C_Builder.build_DRN_C_42(input_shape=(224, 224, 3), num_classes=2)

    #bottleneck 有些问题 params:
    # model1 = DRN_C_Builder.build_DRN_C_58(input_shape=(224, 224, 3), num_classes=2)

    model1.summary()

