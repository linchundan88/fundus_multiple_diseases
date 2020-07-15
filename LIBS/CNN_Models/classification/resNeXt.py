
'''ResNeXt models for Keras.
# Reference
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))
'''

from keras.models import Model
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from LIBS.CNN_Models.classification.MyGroupLayer import MyLayer


#只是增加了 first_filters（原来固定64）和first_kernal_size（原来固定7）的可定义
#原来针对first_filters=64, cardinality=32,width=4
#修改为first_filters=32, cardinality=16,width=4

#first_filters=64  default   cardinality=32, width=4
def my_ResNext(input_shape=None, depth=[2, 3, 4, 5, 3], cardinality=16, width=4, weight_decay=5e-4,
                classes=1000, first_filters=32, first_kernal_size=5):

    img_input = Input(shape=input_shape)

    x = __create_res_next(classes, img_input,  depth, cardinality, width,
                          weight_decay, first_filters=first_filters, first_kernal_size=first_kernal_size)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='resnext')

    return model


def __initial_conv_block(input, weight_decay=5e-4, first_filters=64, first_kernal_size=7):
    ''' Adds an initial conv block, with batch norm and relu for the inception resnext
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    #first_kernal 7
    x = Conv2D(first_filters, (first_kernal_size, first_kernal_size), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), strides=(2, 2))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    return x

def __mygroup(z, c, grouped_channels):  #channel_last
    return z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]

def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = LeakyReLU()(x)
        return x

    for c in range(cardinality):
        #意Lambda 是可以进行参数传递的  字典参数传递  tensorflow Channel last

        x = MyLayer(c=c, grouped_channels=grouped_channels)(input)
        # x = Lambda(__mygroup, arguments={'c': c, 'grouped_channels': grouped_channels})(input)

        # x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        #            if K.image_data_format() == 'channels_last' else
        #            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = LeakyReLU()(x)

    return x


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input

    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = LeakyReLU()(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = LeakyReLU()(x)

    return x


def __create_res_next(nb_classes, img_input,  depth, cardinality=32, width=4,
                      weight_decay=5e-4,  first_filters=64, first_kernal_size=7):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. List of integers.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
    Returns: a Keras Model
    '''

    N = list(depth)

    filters = cardinality * width  #width 的作用
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters

    #开始部分 first conv (7*7) max Pooling
    x = __initial_conv_block(img_input, weight_decay, first_filters=first_filters,
                             first_kernal_size=first_kernal_size)

    # first block
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    filters_list = filters_list[1:]  # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:  #block的第一个conv stride=2
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:   #block中的其他residual unit stride=1
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)


    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)


    return x
