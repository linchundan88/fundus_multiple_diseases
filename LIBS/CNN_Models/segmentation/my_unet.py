from keras.models import Model
from keras.layers import *

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# https://github.com/yihui-he/u-net
# https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
# https://www.kaggle.com/c/ultrasound-nerve-segmentation/data

# https://github.com/seva100/optic-nerve-cnn

# Retina blood vessel segmentation with a convolutional neural network
# https://github.com/orobix/retina-unet

# brain tumor
#https://github.com/zsdonghao/u-net-brain-tumor


'''
注意到，我们这里训练的模型是一个多分类模型，其实更好的做法是，
训练一个二分类模型（使用二分类的标签），对每一类物体进行预测，
得到4张预测图，再做预测图叠加，合并成一张完整的包含4类的预测图，
这个策略在效果上肯定好于一个直接4分类的模型。
所以，U-Net这边我们采取的思路就是对于每一类的分类都训练一个二分类模型，
最后再将每一类的预测结果组合成一个四分类的结果。

那么我们就可以给这些mask图排优先级了，
比如：priority:building>water>road>vegetation，
那么当遇到一个像素点，4个mask图都说是属于自己类别的标签时，
我们就可以根据先前定义好的优先级，把该像素的标签定为优先级最高的标签。

模型融合
多个Mask图，每个像素 多个模型投票表决
'''


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

# the difference between get_unet1 and get_unet2 is only init_filters list_filters
def get_unet1(input_shape, init_filters=64, BN=False, dropout_ratio=0, transpose=False, num_classes=1):
    inputs = Input(shape=input_shape)

    conv1 = _conv_unit(inputs, init_filters, kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = _conv_unit(down1, init_filters * 2, kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = _conv_unit(down2, init_filters * 4, kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = _conv_unit(down3, init_filters * 8, kernel_size=(3, 3), BN=BN, dropout_ratio=dropout_ratio)
    down4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = _conv_unit(down4, init_filters * 16, kernel_size=(3, 3), BN=BN, dropout_ratio=dropout_ratio)

    if transpose:
        upsampling1 = Conv2DTranspose(init_filters*8, (2, 2), strides=(2, 2), padding='same')(center)
    else:
        upsampling1 = UpSampling2D(size=(2, 2))(center)
    up1 = concatenate([upsampling1, conv4], axis=3)
    up1 = _conv_unit(up1, init_filters * 8, kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling2 = Conv2DTranspose(init_filters*4, (2, 2), strides=(2, 2), padding='same')(up1)
    else:
        upsampling2 = UpSampling2D(size=(2, 2))(up1)
    up2 = concatenate([upsampling2, conv3], axis=3)
    up2 = _conv_unit(up2, init_filters * 4, kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling3 = Conv2DTranspose(filters=init_filters*2, kernel_size=(2, 2), strides=(2, 2), padding='same')(up2)
    else:
        upsampling3 = UpSampling2D(size=(2, 2))(up2)
    up3 = concatenate([upsampling3, conv2], axis=3)
    up3 = _conv_unit(up3, init_filters * 2, kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling4 = Conv2DTranspose(init_filters, (2, 2), strides=(2, 2), padding='same')(up3)
    else:
        upsampling4 = UpSampling2D(size=(2, 2))(up3)
    up4 = concatenate([upsampling4, conv1], axis=3)
    up4 = _conv_unit(up4, init_filters, kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    # output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='sigmoid')(up4)

    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(up4)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])

    '''
    final_conv_out = Convolution2D(num_classes, 1, 1)(conv9)
    x = Reshape((num_classes, input_shape[0] * input_shape[1]))(final_conv_out)    
    # x = Permute((2, 1))(x)  #channel first
    
    x = Activation("softmax")(x)  ## output (batch, width*height,label)
    model = Model(inputs=[inputs], outputs=[x])
    
    #keras.layers.Softmax(axis=-1)
    #Output shape  Same shape as the input.

    #softmax：对输入数据的最后一维进行softmax，
    #输入数据应形如(nb_samples, nb_timesteps, nb_dims)
    #或(nb_samples,nb_dims)
    '''

    # model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

    return model

def get_unet2(input_shape, list_filters=[64, 128, 256, 512, 1024], BN=False,
              dropout_ratio=0, transpose=False, num_classes=1):
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

    if transpose:
        upsampling1 = Conv2DTranspose(list_filters[3], (2, 2), strides=(2, 2), padding='same')(center)
    else:
        upsampling1 = UpSampling2D(size=(2, 2))(center)
    up1 = concatenate([upsampling1, conv4], axis=3)
    up1 = _conv_unit(up1, list_filters[3], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling2 = Conv2DTranspose(list_filters[2], (2, 2), strides=(2, 2), padding='same')(up1)
    else:
        upsampling2 = UpSampling2D(size=(2, 2))(up1)
    up2 = concatenate([upsampling2, conv3], axis=3)
    up2 = _conv_unit(up2,list_filters[2], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling3 = Conv2DTranspose(filters=list_filters[1], kernel_size=(2, 2), strides=(2, 2), padding='same')(up2)
    else:
        upsampling3 = UpSampling2D(size=(2, 2))(up2)
    up3 = concatenate([upsampling3, conv2], axis=3)
    up3 = _conv_unit(up3, list_filters[1], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling4 = Conv2DTranspose(list_filters[0], (2, 2), strides=(2, 2), padding='same')(up3)
    else:
        upsampling4 = UpSampling2D(size=(2, 2))(up3)
    up4 = concatenate([upsampling4, conv1], axis=3)
    up4 = _conv_unit(up4, list_filters[0], kernel_size=(3, 3), BN=BN, dropout_ratio=0)


    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(up4)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def get_unet_small(input_shape, list_filters=[32, 64, 128, 256], BN=False,
              dropout_ratio=0, transpose=False, num_classes=1):
    inputs = Input(shape=input_shape)

    conv1 = _conv_unit(inputs, list_filters[0], kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = _conv_unit(down1, list_filters[1], kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = _conv_unit(down2, list_filters[2], kernel_size=(3, 3), BN=BN, dropout_ratio=0)
    down3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    center = _conv_unit(down3, list_filters[3], kernel_size=(3, 3), BN=BN, dropout_ratio=dropout_ratio)

    if transpose:
        upsampling1 = Conv2DTranspose(list_filters[2], (2, 2), strides=(2, 2), padding='same')(center)
    else:
        upsampling1 = UpSampling2D(size=(2, 2))(center)
    up1 = concatenate([upsampling1, conv3], axis=3)
    up1 = _conv_unit(up1, list_filters[3], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling2 = Conv2DTranspose(list_filters[1], (2, 2), strides=(2, 2), padding='same')(up1)
    else:
        upsampling2 = UpSampling2D(size=(2, 2))(up1)
    up2 = concatenate([upsampling2, conv2], axis=3)
    up2 = _conv_unit(up2, list_filters[2], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    if transpose:
        upsampling3 = Conv2DTranspose(filters=list_filters[1], kernel_size=(2, 2), strides=(2, 2), padding='same')(up2)
    else:
        upsampling3 = UpSampling2D(size=(2, 2))(up2)
    up3 = concatenate([upsampling3, conv1], axis=3)
    up3 = _conv_unit(up3, list_filters[0], kernel_size=(3, 3), BN=BN, dropout_ratio=0)

    output = Conv2D(filters=num_classes, kernel_size=(1, 1))(up3)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[inputs], outputs=[output])

    return model


if __name__ == '__main__':
    print('start')
    # model = get_unet1( input_shape=(384, 384, 3))
    # plot_model(model, to_file='MobileNetv2.png', show_shapes=True)
    # model.summary()
    # model.layers[48].output.shape  (-1,384,384,64)  # channel last
    # model.layers[49].output.shape  (-1,384,384,1)   # conv 1*1