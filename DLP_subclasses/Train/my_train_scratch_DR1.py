
import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
GPU_NUM = 3
import pandas as pd
import numpy as np
from LIBS.DataPreprocess import my_data
from LIBS.DataPreprocess.my_images_generator import My_images_generator, My_images_weight_generator
import keras.backend as K
import keras.optimizers
from keras.callbacks import ModelCheckpoint
import math, collections
from LIBS.CNN_Models import my_transfer_learning
from LIBS.CNN_Models.my_multi_gpu import ModelMGPU


def train_task(filename_csv_train, filename_csv_valid,
               epoch_fine_tuning=20):

    str_subclass_no = '0.3'
    TRAIN_TYPE = 'DLP_SubClass' + str_subclass_no
    IMG_AUG_ROTATE_MODE = 1 #1:do flip,roate, 2:do flip 3: only translate_percent
    MODEL_SAVE_DIR = '/tmp2/models_subclass/' + TRAIN_TYPE

    #region read csv set weight_class_start, split train validation set
    df = pd.read_csv(filename_csv_train)
    NUM_CLASSES = df['labels'].nunique(dropna=True)

    #  len(df.loc[(df['labels'] == 0)])
    weight_class_start = np.array([1, 9.5])  #23648, 2020
    weight_class_end = np.array([1, 9.5])
    balance_ratio = 0.93

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(filename_csv_valid, shuffle=False)

    #endregion

    #region load pre-trained modal

    model_name = 'ResNet448'
    IMAGE_SIZE = 448
    # model_file = '/home/ubuntu/dlp/deploy_models_2019/SubClass0_3/ResNet448-008-train0.7611_val0.840.hdf5'
    model_file = '/home/ubuntu/dlp/deploy_models/DR0_DR1/ResNet448-007-train0.8310_val0.883.hdf5'

    model1 = keras.models.load_model(model_file, compile=False)
    model1.summary()

    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 64

    #endregion

    #region save model dir and checkpointer

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_save_filepath = os.path.join(MODEL_SAVE_DIR, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")

    checkpointer = ModelCheckpoint(model_save_filepath, verbose=1,
                                   save_weights_only=False, save_best_only=False)
    #endregion

    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


    #region train header layers

    model_train_top = my_transfer_learning.convert_model_transfer(model1, change_top=False, clsss_num=NUM_CLASSES)

    #endregion

    #region change learn rate(only fine tuning )

    def scheduler(epoch):
        try:
            file_object = open('lr.txt')
            line = file_object.readline()
            file_object.close()
            line = line.strip('\n') #删除换行符
            lr_rate = float(line)

            print('set learning rate by lr.txt')
            print("epoch：%d, current learn rate:  %f" % (epoch, lr_rate))
            K.set_value(model1.optimizer.lr, lr_rate)

        except Exception:
            print('read lr-rate file error')
            print('set learning rate automatically')

            # 内置dictionary数据类型是无序的
            dict_lr_rate = collections.OrderedDict()
            dict_lr_rate['0'] = 1e-3  # 0.00001
            dict_lr_rate['2'] = 2e-4
            dict_lr_rate['4'] = 1e-4
            dict_lr_rate['6'] = 3e-5
            dict_lr_rate['8'] = 1e-5
            dict_lr_rate['10'] = 1e-6  # 0.000001
            dict_lr_rate['15'] = 6e-7

            for (k, v) in dict_lr_rate.items():
                if epoch >= int(k):
                    lr_rate = v
            print("epoch：%d, current learn rate:  %f" % (epoch, lr_rate))
            K.set_value(model1.optimizer.lr, lr_rate)

        return K.get_value(model1.optimizer.lr)

    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    # endregion

    #region fine tuning all layers

    model_fine_tune = my_transfer_learning.convert_trainable_all(model_train_top)
    if GPU_NUM > 1:
        print('convert fine tuning model to Multiple GPU...')
        model1 = ModelMGPU(model_fine_tune, GPU_NUM)
        print('convert fine tuning model to Multiple GPU OK')
    else:
        model1 = model_fine_tune

    op_adam_fine_tune = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model1.compile(loss='categorical_crossentropy',
                   optimizer=op_adam_fine_tune, metrics=['acc'], weighted_metrics=['acc'])

    #region data generator

    from imgaug import augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    # sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
    imgaug_train = iaa.Sequential([
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.04, 0.04),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.2),  # horizontally flip 50% of the images
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        # shuortcut for CropAndPad

        # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
        # sometimes1(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5), ),
        # change brightness of images (by -5 to 5 of original value)
        # sometimes1(iaa.Add((-6, 6), per_channel=0.5),),
        sometimes(iaa.Affine(
            # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
            # scale images to 80-120% of their size, individually per axis
            # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),  # rotate by -10 to +10 degrees
            # shear=(-16, 16),  # shear by -16 to +16 degrees
            # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    ])


    my_gen_train = My_images_weight_generator(files=train_files, labels=train_labels, image_shape=image_shape,
                                              weight_class_start=weight_class_start, weight_class_end=weight_class_end, balance_ratio=balance_ratio,
                                              num_class=NUM_CLASSES, imgaug_seq=imgaug_train, batch_size=BATCH_SIZE_TRAIN)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels, image_shape=image_shape,
                                       num_output=NUM_CLASSES, batch_size=BATCH_SIZE_VALID)

    #endregion

    history_fine_tuning = model1.fit_generator(
        my_gen_train.gen(),
        steps_per_epoch=math.ceil(len(train_files) / BATCH_SIZE_TRAIN), #number of training batch
        epochs=epoch_fine_tuning,
        validation_data=my_gen_valid.gen(),
        validation_steps=math.ceil(len(valid_files) / BATCH_SIZE_VALID),
        callbacks=[checkpointer, change_lr]
    )

    #endregion

    K.clear_session()  #release GPU memory


# filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
#                             'datafiles', 'Subclass_0.3.csv'))

filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                                'datafiles', 'Subclass_0.3_train.csv'))
filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                                  'datafiles', 'Subclass_0.3_valid.csv'))


train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
           epoch_fine_tuning=20)




print('OK!')

