'''
    3D CNN
'''

from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation
from keras.layers import Dropout, Input, BatchNormalization,GlobalAveragePooling3D
from keras.models import Model

def get_3d_model1(input_shape=(128, 128, 128, 1), NUM_CLASSES=2):
    ## input layer
    input_layer = Input(input_shape)

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer1)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    conv_layer5 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer2)
    conv_layer6 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer5)
    conv_layer7 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer6)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer7)

    conv_layer8 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer3)
    conv_layer9 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer8)
    conv_layer10 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer9)
    pooling_layer4 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer10)


    flatten_layer = Flatten()(pooling_layer4)

    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=64, activation='relu')(dense_layer1)
    # dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def get_3d_model2(input_shape=(128, 128, 128, 1), NUM_CLASSES=2):
    ## input layer
    input_layer = Input(input_shape)

    ## convolutional layers
    conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer1)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    conv_layer5 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer2)
    conv_layer6 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer5)
    conv_layer7 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer6)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer7)

    conv_layer8 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu')(pooling_layer3)
    conv_layer9 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer8)
    conv_layer10 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv_layer9)
    pooling_layer4 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer10)

    flatten_layer = Flatten()(pooling_layer4)

    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    # dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=256, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def my_conv3d(input, filters=32, kernal_size=3, strides=1, padding='same'):
    x = Conv3D(filters=filters, kernel_size=kernal_size, strides=strides,
               padding=padding)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_3d_model3(input_shape=(64, 64, 64, 1), NUM_CLASSES=2):
    ## input layer
    input_layer = Input(input_shape)

    ## convolutional layers

    conv_layer1 = my_conv3d(input_layer, filters=32, kernal_size=(7, 7, 7), strides=2, padding='same')

    conv_layer2 = my_conv3d(conv_layer1, filters=32, kernal_size=(5, 5, 5), padding='same')

    conv_layer3 = my_conv3d(conv_layer2, filters=32, kernal_size=(3, 3, 3), padding='same')

    conv_layer4 = my_conv3d(conv_layer3, filters=32, kernal_size=(3, 3, 3), padding='same')

    conv_layer5 = my_conv3d(conv_layer4, filters=32, kernal_size=(3, 3, 3), padding='same')


    GAP = GlobalAveragePooling3D(name='avg_pool')(conv_layer5)

    output_layer = Dense(units=NUM_CLASSES, activation='softmax')(GAP)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)


    return model