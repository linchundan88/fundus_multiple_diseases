
import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
GPU_NUM = 2
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

MODEL_SAVE_DIR = '/tmp2/models_subclass/'

# model_no=1 (Xception)  model_no=2 (Inception-ResnetV2)
def train_task(filename_csv_train, filename_csv_valid, sub_class_no, model_no,
               epoch_header=None, epoch_fine_tuning=None):

    str_subclass_no = str(sub_class_no)
    TRAIN_TYPE = 'DLP_SubClass' + str_subclass_no
    IMG_AUG_ROTATE_MODE = 1 #1:do flip,roate, 2:do flip 3: only translate_percent
    model_save_dir = MODEL_SAVE_DIR + TRAIN_TYPE

    #region read csv set weight_class_start, split train validation set
    df = pd.read_csv(filename_csv_train)
    NUM_CLASSES = df['labels'].nunique(dropna=True)

    #  len(df.loc[(df['labels'] == 0)])
    if sub_class_no == 0.1:  #Tessellated fundus  4567, 1324
        weight_class_start = np.array([1, 2.4])
        weight_class_end = np.array([1, 2.4])
        balance_ratio = 0.93
    if sub_class_no == 0.2:  # Big Optic Cup 4567,6639
        weight_class_start = np.array([1, 0.5])
        weight_class_end = np.array([1, 0.5])
        balance_ratio = 0.93
    if sub_class_no == 1:  #DR2,3  10284,2490, single label:9715, 2636
        # weight_class_start = np.array([1, 2.4])
        # weight_class_end = np.array([1, 2.4])
        #12949,3129,  8412,2241
        #add test dataset  11156,2664
        weight_class_start = np.array([1, 2.9])
        weight_class_end = np.array([1, 2.9])
        balance_ratio = 0.93
    if sub_class_no == 2:  #Data CRVO  2636,1527, single label:2548,1391
        weight_class_start = np.array([1, 1.4])
        weight_class_end = np.array([1, 1.4])
        balance_ratio = 0.93
    if sub_class_no == 5:   #543, 683, single label:665,604
        weight_class_start = np.array([1, 1])
        weight_class_end = np.array([1, 1])
        balance_ratio = 0.93
    if sub_class_no == 10:   # 5953 1449
        weight_class_start = np.array([1, 2.6])
        weight_class_end = np.array([1, 2.6])
        balance_ratio = 0.93
    if sub_class_no == 15:  # single label:1523,136
        weight_class_start = np.array([1, 6])
        weight_class_end = np.array([1, 6])
        balance_ratio = 0.93
    if sub_class_no == 29:  # Blur 16253 1814, single label:20882,1099, 12949,3129,
        #train:17580, 1234, total:20709,1432
        weight_class_start = np.array([1, 11])
        weight_class_end = np.array([1, 11])
        balance_ratio = 0.93

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(filename_csv_valid, shuffle=False)
    #endregion

    #region load pre-trained modal
    if model_no == 1:
        model_name = 'Xception'
        IMAGE_SIZE = 299
        model_file = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_Xception-015-train0.9671_val0.945.hdf5'
    elif model_no == 2:
        model_name = 'InceptionResNetV2'
        IMAGE_SIZE = 299
        model_file = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_InceptionResNetV2-006-train0.9674_val0.951.hdf5'
    elif model_no == 3:
        model_name = 'InceptionV3'
        IMAGE_SIZE = 299
        model_file = '/home/ubuntu/dlp/deploy_models/bigclasses/2018_5_3/InceptionV3-008-train0.9631_val0.9635.hdf5'

    model1 = keras.models.load_model(model_file, compile=False)

    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 64

    #endregion

    #region save model dir and checkpointer

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_filepath = os.path.join(model_save_dir, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")

    checkpointer = ModelCheckpoint(model_save_filepath, verbose=1,
                                   save_weights_only=False, save_best_only=False)
    #endregion


    #region data generator
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    from imgaug import augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    # sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
    imgaug_train_seq = iaa.Sequential([
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
                                              num_class=NUM_CLASSES, imgaug_seq=imgaug_train_seq, batch_size=BATCH_SIZE_TRAIN)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels, image_shape=image_shape,
                                       num_output=NUM_CLASSES, batch_size=BATCH_SIZE_VALID)

    #endregion

    #region train header layers

    model_train_top = my_transfer_learning.convert_model_transfer(model1, change_top=True, clsss_num=NUM_CLASSES)
    if GPU_NUM > 1:
        print('convert base model to Multiple GPU...')
        model1 = ModelMGPU(model_train_top, GPU_NUM)
        print('convert base top model to Multiple GPU OK')
    else:
        model1 = model_train_top

    op_adam_train_top = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_train_top.compile(loss='categorical_crossentropy',
                   optimizer=op_adam_train_top, metrics=['acc'], weighted_metrics=['acc'])

    if epoch_header is None:
        if len(df) > 10000:
            epoch_header = 5
        elif len(df) > 5000:
            epoch_header = 8
        elif len(df) > 2000:
            epoch_header = 10
        else:
            epoch_header = 15

    history_top = model_train_top.fit_generator(
        my_gen_train.gen(),
        steps_per_epoch=math.ceil(len(train_files) / BATCH_SIZE_TRAIN), #number of training batch
        epochs=epoch_header,
        validation_data=my_gen_valid.gen(),
        validation_steps=math.ceil(len(valid_files) / BATCH_SIZE_VALID),
    )

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
            dict_lr_rate['0'] = 3e-5
            dict_lr_rate['3'] = 1e-5  # 0.00001
            dict_lr_rate['6'] = 2e-6
            dict_lr_rate['9'] = 1e-6  # 0.000001
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
        print('convert fine-tuning model to Multiple GPU...')
        model_fine_tune = ModelMGPU(model_fine_tune, GPU_NUM)
        print('convert fine-tuning model to Multiple GPU OK')

    op_adam_fine_tune = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model_fine_tune.compile(loss='categorical_crossentropy',
                   optimizer=op_adam_fine_tune, metrics=['acc'], weighted_metrics=['acc'])

    if epoch_fine_tuning is None:
        if len(df) > 10000:
            epoch_fine_tuning = 20
        elif len(df) > 5000:
            epoch_fine_tuning = 25
        elif len(df) > 2000:
            epoch_fine_tuning = 30
        else:
            epoch_fine_tuning = 40

    history_fine_tuning = model_fine_tune.fit_generator(
        my_gen_train.gen(),
        steps_per_epoch=math.ceil(len(train_files) / BATCH_SIZE_TRAIN), #number of training batch
        epochs=epoch_fine_tuning,
        validation_data=my_gen_valid.gen(),
        validation_steps=math.ceil(len(valid_files) / BATCH_SIZE_VALID),
        callbacks=[checkpointer, change_lr]
    )

    #endregion

    K.clear_session()  #release GPU memory


# for subclass_no in [0.1, 0.2, 1, 2, 5, 15, 29]:
for subclass_no in [1]:
    filename_csv_subclass = os.path.abspath(
        os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_{0}.csv'.format(subclass_no)))
    # DLP_SubClass_1_new.csv
    filename_csv_subclass = os.path.abspath(
        os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_1_new.csv'))

    # filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
    #                                         'datafiles', 'Subclass_0.3.csv'))

    # filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
    #                                         'datafiles', 'Subclass_1.csv'))

    # filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
    #                                             'datafiles', 'Subclass_1_train.csv'))
    # filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
    #                                                   'datafiles', 'Subclass_1_valid.csv'))

    filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                                                'datafiles', 'Subclass_1_train_add_test.csv'))
    filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                                                  'datafiles', 'Subclass_1_valid_add_test.csv'))


    train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
               sub_class_no=subclass_no, model_no=1)
    train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
               sub_class_no=subclass_no, model_no=2)

# train_task(sub_class_no=0.1, model_no=1)
# train_task(sub_class_no=0.1, model_no=2)


print('OK!')

