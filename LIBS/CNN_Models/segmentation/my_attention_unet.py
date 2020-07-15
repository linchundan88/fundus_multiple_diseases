'''
  Paper:Attention U-Net: Learning Where to Look for the Pancreas
  # https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64367

'''

import keras
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, BatchNormalization, \
    Activation, add, multiply
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D


def _conv_unit(input, filters, kernel_size=(3, 3), BN=False, dropout_ratio=0):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input)
    if BN:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_ratio > 0:
        x = Dropout(dropout_ratio)(x)

    #repeat 2 times
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    if BN:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_ratio > 0:
        x = Dropout(dropout_ratio)(x)

    return x

def up_conv(input, filters , kernel_size=(3, 3), transpose=False):
    if transpose:
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input)
    else:
        x = UpSampling2D(size=(2, 2))(input)


    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def AttentionBlock(g, x, filters):
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


def get_attention_unet(input_shape,  list_filters=[64, 128, 256, 512, 1024],
           BN=False, transpose=False, dropout_ratio=0, num_classes=1, activation='sigmoid'):

    inputs = Input(shape=input_shape)

    conv1 = _conv_unit(inputs, list_filters[0], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    down1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = _conv_unit(down1, list_filters[1], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    down2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = _conv_unit(down2, list_filters[2], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    down3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = _conv_unit(down3, list_filters[3], kernel_size=(3, 3), BN=BN, dropout_ratio=dropout_ratio)

    down4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = _conv_unit(down4, list_filters[4], kernel_size=(3, 3), BN=BN, dropout_ratio=dropout_ratio)

    g1 = up_conv(input=center, filters=list_filters[3])
    atten1 = AttentionBlock(g=g1, x=conv4, filters=list_filters[3]//2)
    up1 = concatenate([atten1, g1], axis=3)
    up1 = _conv_unit(up1, list_filters[3], kernel_size=(3, 3), BN=BN, transpose=transpose, dropout_ratio=0)

    g2 = up_conv(input=up1, filters=list_filters[2])
    atten2 = AttentionBlock(g=g2, x=conv3, filters=list_filters[2]//2)
    up2 = concatenate([atten2, g2], axis=3)
    up2 = _conv_unit(up2, list_filters[2], kernel_size=(3, 3), BN=BN, transpose=transpose, dropout_ratio=0)

    g3 = up_conv(input=up2, filters=list_filters[1])
    atten3 = AttentionBlock(g=g3, x=conv2, filters=list_filters[1]//2)
    up3 = concatenate([atten3, g3], axis=3)
    up3 = _conv_unit(up3, list_filters[1], kernel_size=(3, 3), BN=BN, transpose=transpose, dropout_ratio=0)

    g4 = up_conv(input=up3, filters=list_filters[0])
    atten4 = AttentionBlock(g=g4, x=conv1, filters=list_filters[0]//2)
    up4 = concatenate([atten4, g4], axis=3)
    up4 = _conv_unit(up4, list_filters[0], kernel_size=(3, 3), BN=BN, transpose=transpose, dropout_ratio=0)

    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(up4)
    output = Activation(activation)(output) # output = Activation('sigmoid')(output)


    model = Model(inputs=[inputs], outputs=[output])

    return model


if __name__ == '__main__':

    model = get_attention_unet(input_shape=(384, 384, 3))

    model.summary()
    # model.layers[48].output.shape  (-1,384,384,64)  # channel last
    # model.layers[49].output.shape  (-1,384,384,1)
