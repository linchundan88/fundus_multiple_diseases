
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
GPU_NUM = 2
import pandas as pd
import numpy as np
from LIBS.DataPreprocess import my_data
from LIBS.DataPreprocess.my_images_generator import my_Generator, My_images_weight_generator
import keras.backend as K
import keras.optimizers
from keras.callbacks import ModelCheckpoint
import math, collections
from LIBS.CNN_Models import my_transfer_learning
from LIBS.CNN_Models.my_multi_gpu import ModelMGPU
from LIBS.DataPreprocess.my_data import write_csv_based_on_dir

#region set parameters and split data
GEN_CSV = True

TRAIN_TYPE = 'DLP_SubClass10'
MODEL_SAVE_DIR = '/tmp2/models_subclass/' + TRAIN_TYPE
img_aug_mode = 1  # flip,roate
dir_preprocess = '/media/ubuntu/data1/big_class_sub_classes/crop_optic_disc/10'
dir_preprocess = '/tmp2/SubClass10_new/Crop_optic_disc_112/10/'
# filename_csv = os.path.join(sys.path[0], 'datafiles', TRAIN_TYPE + '.csv')
filename_csv = TRAIN_TYPE + '.csv'
dict_mapping = {'0': 0, '1': 1}

if GEN_CSV:
    write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping)

df = pd.read_csv(filename_csv)  # in order to determine class sample size
NUM_CLASSES = df['labels'].nunique(dropna=True)

#  len(df.loc[(df['labels'] == 0)])   # 1192, 647.  1311, 1120
weight_class_start = np.array([1, 1])
weight_class_end = np.array([1, 1])
balance_ratio = 0.93

train_files, train_labels, valid_files, valid_labels = my_data.split_dataset(
    filename_csv, valid_ratio=0.15, random_state=1111)

from LIBS.DataPreprocess.my_data import write_images_labels_csv
write_images_labels_csv(train_files, train_labels, filename_csv=TRAIN_TYPE + '_train.csv')
write_images_labels_csv(valid_files, valid_labels, filename_csv=TRAIN_TYPE + '_valid.csv')

#endregion


def train_task(model_name, model_file, image_size, dict_lr_rate=None,
       change_top=True, epoch_top=2, epoch_finetuning=15,
       batch_size_train=32, batch_size_valid=32):

    #region load pre_trained Model,  model_name(callback model save will use)

    image_shape = (image_size, image_size, 3)

    print('load pre-trained model...')
    #mobilenetv2
    from keras.utils.generic_utils import CustomObjectScope
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        model1 = keras.models.load_model(model_file, compile=False)
    print('load pre-trained model OK')

    #endregion

    #region callback function savemodel, change learn rate

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_save_filepath = os.path.join(MODEL_SAVE_DIR, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")

    checkpointer = ModelCheckpoint(model_save_filepath, verbose=1,
                   save_weights_only=False, save_best_only=False)

    #change lr-rate only for fine-tuning
    if dict_lr_rate is None:
        dict_lr_rate = collections.OrderedDict()
        dict_lr_rate['0'] = 1e-5  # 0.00001
        dict_lr_rate['3'] = 3e-6  # 0.000003
        dict_lr_rate['5'] = 1e-6  # 0.000001
        dict_lr_rate['9'] = 6e-7

    def scheduler(epoch):
        try:
            file_object = open('lr.txt')
            line = file_object.readline()
            file_object.close()
            line = line.strip('\n') #删除换行符
            lr_rate = float(line)

            print('set learning rate by lr.txt')
        except Exception:
            print('read lr-rate file error')
            print('set learning rate automatically')

            for (k, v) in dict_lr_rate.items():
                if epoch >= int(k):
                    lr_rate = v

        print("epoch：%d, current learn rate:  %f" % (epoch, lr_rate))
        K.set_value(model1.optimizer.lr, lr_rate)

        return K.get_value(model1.optimizer.lr)

    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    # endregion

    #region  data generator implement dynamic resampling

    my_gen = My_images_weight_generator(files=train_files, labels=train_labels,
                                        weight_class_start=weight_class_start, weight_class_end=weight_class_end,
                                        balance_ratio=balance_ratio,
                                        num_class=NUM_CLASSES, img_aug_mode=img_aug_mode,
                                        batch_size=batch_size_train, image_shape=image_shape)
    #endregion


    #region cnvert model(top layer) and  train top layers(if epoch_top >0)
    model_train_top = my_transfer_learning.convert_model_transfer(model1, change_top=change_top,
                                                                  clsss_num=NUM_CLASSES, activation_function='SoftMax', freeze_feature_extractor=True)

    if GPU_NUM > 1:
        print('convert base model to Multiple GPU...')
        model1 = ModelMGPU(model_train_top, GPU_NUM)
        print('convert base model to Multiple GPU OK')
    else:
        model1 = model_train_top

    if epoch_top > 0:
        op_adam_train_top = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model1.compile(loss='categorical_crossentropy',
                   optimizer=op_adam_train_top, metrics=['acc'])

        history_top = model1.fit_generator(
            my_gen.gen(),
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train), #number of training batch
            epochs=epoch_top,
            validation_data=my_Generator(valid_files, valid_labels,
                 image_shape=image_shape, batch_size=batch_size_valid,
                 num_class=NUM_CLASSES, train_or_valid='valid'),
            validation_steps=math.ceil(len(valid_files) / batch_size_valid)
        )

    #endregion

    #region fine tuning all layers

    model_fine_tune = my_transfer_learning.convert_trainable_all(model_train_top)

    if GPU_NUM > 1:
        print('convert fine-tuning model to Multiple GPU...')
        model_fine_tune = ModelMGPU(model_fine_tune, GPU_NUM)
        print('convert fine-tuning model to Multiple GPU OK')

    op_adam_fine_tune = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model1.compile(loss='categorical_crossentropy',
                   optimizer=op_adam_fine_tune, metrics=['acc'])

    history_fine_tuning = model_fine_tune.fit_generator(
        my_gen.gen(),
        steps_per_epoch=math.ceil(len(train_files) / batch_size_train), #number of training batch
        epochs=epoch_finetuning,
        validation_data=my_Generator(valid_files, valid_labels,
                 image_shape=image_shape, batch_size=batch_size_valid,
                 num_class=NUM_CLASSES, train_or_valid='valid'),
        validation_steps=math.ceil(len(valid_files) / batch_size_valid),
        callbacks=[checkpointer, change_lr]
    )

    #endregion

    K.clear_session()  #release GPU memory


dict_lr_rate = collections.OrderedDict()
dict_lr_rate['0'] = 3e-4
dict_lr_rate['4'] = 1e-4   #0.0001
dict_lr_rate['8'] = 1e-5
dict_lr_rate['12'] = 3e-6
dict_lr_rate['20'] = 1e-6
dict_lr_rate['30'] = 6e-7


BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64

model_name = 'Resnet112'
IMAGE_SIZE3 = 112
# model_file1 = '/home/ubuntu/dlp/deploy_models_2019/SubClass10/Resnet112-024-0.953.hdf5'
# model_file1 = '/home/ubuntu/dlp/deploy_models_new/Subclass11/Resnet-038-0.932.hdf5'
model_file1 = '/home/ubuntu/dlp/deploy_models_2019/SubClass10_new/Resnet112-030-0.866.hdf5'
train_task(model_name, model_file1, image_size=IMAGE_SIZE3, dict_lr_rate=dict_lr_rate,
           change_top=False, epoch_top=5, epoch_finetuning=50,
           batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_VALID)


print('OK')
