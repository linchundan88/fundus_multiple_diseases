'''
U-net with simple Resnet Blocks
https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
'''

from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
    Dropout, Cropping2D, Add


def _BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def _convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = _BatchActivate(x)
    return x

def _residual_block(blockInput, num_filters=16, batch_activate = False):
    x = _BatchActivate(blockInput)
    x = _convolution_block(x, num_filters, (3, 3))
    x = _convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = _BatchActivate(x)
    return x

ACTIVATION = "relu"

# Build model
def get_unet_resnet_v2(input_shape, init_filters=16, dropout_ratio=0.2):

    input_layer = Input(shape=input_shape)

    # 101 -> 50
    conv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = _residual_block(conv1, init_filters * 1)
    conv1 = _residual_block(conv1, init_filters * 1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_ratio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = _residual_block(conv2, init_filters * 2)
    conv2 = _residual_block(conv2, init_filters * 2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_ratio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = _residual_block(conv3, init_filters * 4)
    conv3 = _residual_block(conv3, init_filters * 4)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_ratio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(init_filters * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = _residual_block(conv4, init_filters * 8)
    conv4 = _residual_block(conv4, init_filters * 8)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_ratio)(pool4)

    # Middle
    convm = Conv2D(init_filters * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = _residual_block(convm, init_filters * 16)
    convm = _residual_block(convm, init_filters * 16)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(init_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_ratio)(uconv4)

    uconv4 = Conv2D(init_filters * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = _residual_block(uconv4, init_filters * 8)
    uconv4 = _residual_block(uconv4, init_filters * 8)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(init_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_ratio)(uconv3)

    uconv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = _residual_block(uconv3, init_filters * 4)
    uconv3 = _residual_block(uconv3, init_filters * 4)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(init_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(dropout_ratio)(uconv2)
    uconv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = _residual_block(uconv2, init_filters * 2)
    uconv2 = _residual_block(uconv2, init_filters * 2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(init_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(dropout_ratio)(uconv1)
    uconv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = _residual_block(uconv1, init_filters * 1)
    uconv1 = _residual_block(uconv1, init_filters * 1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # uconv1 = Dropout(dropout_ratio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model



# Build model
def get_unet_resnet_v2_small(input_shape, init_filters=32, dropout_ratio=0.2):

    input_layer = Input(shape=input_shape)

    # 101 -> 50
    conv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = _residual_block(conv1, init_filters * 1)
    conv1 = _residual_block(conv1, init_filters * 1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_ratio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = _residual_block(conv2, init_filters * 2)
    conv2 = _residual_block(conv2, init_filters * 2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_ratio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = _residual_block(conv3, init_filters * 4)
    conv3 = _residual_block(conv3, init_filters * 4)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_ratio)(pool3)

    # Middle
    convm = Conv2D(init_filters * 16, (3, 3), activation=None, padding="same")(pool3)
    convm = _residual_block(convm, init_filters * 16)
    convm = _residual_block(convm, init_filters * 16)
    convm = Activation(ACTIVATION)(convm)


    # 12 -> 25
    deconv3 = Conv2DTranspose(init_filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_ratio)(uconv3)

    uconv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = _residual_block(uconv3, init_filters * 4)
    uconv3 = _residual_block(uconv3, init_filters * 4)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(init_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(dropout_ratio)(uconv2)
    uconv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = _residual_block(uconv2, init_filters * 2)
    uconv2 = _residual_block(uconv2, init_filters * 2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(init_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(dropout_ratio)(uconv1)
    uconv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = _residual_block(uconv1, init_filters * 1)
    uconv1 = _residual_block(uconv1, init_filters * 1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    # uconv1 = Dropout(dropout_ratio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model