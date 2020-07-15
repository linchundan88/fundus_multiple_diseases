import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model


def DenseBlock(inputs, outdim):
    inputshape = K.int_shape(inputs)
    bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                          beta_initializer='zero', gamma_initializer='one')(inputs)
    act = Activation('relu')(bn)
    conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

    if inputshape[3] != outdim:
        shortcut = Conv2D(outdim, (1, 1), padding='same')(inputs)
    else:
        shortcut = inputs
    result1 = add([conv1, shortcut])

    bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                          beta_initializer='zero', gamma_initializer='one')(result1)
    act = Activation('relu')(bn)
    conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
    result = add([result1, conv2, shortcut])
    result = Activation('relu')(result)

    return result


def my_get_dense_unet(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
    conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                             beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = DenseBlock(conv1, 32)  # 48
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = DenseBlock(pool1, 64)  # 24
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = DenseBlock(pool2, 128)  # 12
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = DenseBlock(pool3, 256)  # 12

    up1 = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv4)
    up1 = concatenate([up1, conv3], axis=3)

    conv5 = DenseBlock(up1, 64)

    up2 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv5)
    up2 = concatenate([up2, conv2], axis=3)

    conv6 = DenseBlock(up2, 64)

    up3 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(conv6)
    up3 = concatenate([up3, conv1], axis=3)

    conv7 = DenseBlock(up3, 32)

    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(conv7)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def my_get_dense_unet1(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
    conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                                             beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = DenseBlock(conv1, 32)  # 48
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = DenseBlock(pool1, 64)  # 24
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = DenseBlock(pool2, 64)  # 12
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = DenseBlock(pool3, 64)  # 12

    up1 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv4)
    up1 = concatenate([up1, conv3], axis=3)

    conv5 = DenseBlock(up1, 64)

    up2 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv5)
    up2 = concatenate([up2, conv2], axis=3)

    conv6 = DenseBlock(up2, 64)

    up3 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(conv6)
    up3 = concatenate([up3, conv1], axis=3)

    conv7 = DenseBlock(up3, 32)

    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(conv7)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])


    return model



if __name__ == '__main__':
    print('start')
    # model = get_unet1( input_shape=(384, 384, 3))
    # plot_model(model, to_file='MobileNetv2.png', show_shapes=True)
