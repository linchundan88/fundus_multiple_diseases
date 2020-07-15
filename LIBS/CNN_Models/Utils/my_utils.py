
from keras.optimizers import *
from keras.layers import *


# get the last conv layer number
def get_last_conv_layer_number(model):
    #
    for i in range(len(model.layers)-1, -1, -1):
        if isinstance(model.layers[i], Conv2D) or \
                isinstance(model.layers[i], Activation) or \
                isinstance(model.layers[i], SeparableConv2D) or\
                isinstance(model.layers[i], Concatenate) :  #inception v3 Concatenate

            last_conv_layer = i
            break

    return last_conv_layer

# get the last conv layer name
def get_last_conv_layer_name(model):
    layer_number = get_last_conv_layer_number(model)

    return model.layers[layer_number].name


def get_GAP_layer_num(model):

    # get the global average pooling layer

    for i in range(len(model.layers)-1, -1, -1):
        if isinstance(model.layers[i], GlobalAveragePooling2D) or \
                isinstance(model.layers[i], AveragePooling2D):
            GAP_layer = i
            break

    return GAP_layer



def interpolation(input_tensor, ref_tensor, name):  # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value], name=name)

def interpolation1(input_tensor, ref_tensor):  # resizes input_tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])



#  g and x have different dim 输出和x的维度 一样
def AttentionBlock(g, x, filters):
    if (g.shape[-3] != x.shape[-3]):
        g = Lambda(interpolation1, arguments={'ref_tensor': x})(g)
        # g = interpolation(g, x, 'upsampling_g')

    g1 = Conv2D(filters, kernel_size=1)(g)
    g1 = BatchNormalization()(g1)

    x1 = Conv2D(filters, kernel_size=1)(x)
    x1 = BatchNormalization()(x1)

    g1_x1 = add([g1, x1])
    psi = Activation('relu')(g1_x1)

    psi = Conv2D(1, kernel_size=1)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    x = multiply([x, psi])

    return x

