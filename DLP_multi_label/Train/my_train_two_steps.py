'''
  New multi-labels data1
  New Methods:
     dynamic resampling based on verified labels
         (change from multi-labels to single label)
     computing labels instead of on-hot encoding)
'''

import sys, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
GPU_NUM = 3

import pandas as pd
from LIBS.DataPreprocess import my_data
from LIBS.DataPreprocess.my_images_generator import My_images_generator, My_gen_weight_multi_labels
from LIBS.DataPreprocess.my_images_generator import op_class_weight
import keras.backend as K
import keras.optimizers
from keras.callbacks import ModelCheckpoint
import math, collections
import numpy as np
from LIBS.CNN_Models.my_loss.my_metrics import get_weighted_binary_crossentropy
from LIBS.CNN_Models import my_transfer_learning
from LIBS.CNN_Models.my_multi_gpu import ModelMGPU


def train_task(filename_csv_train, filename_csv_valid, num_classes,
               model_name, model1, image_size,
               add_top=False, change_top=True, freeze_layes_num=None,
               class_weights=None, inter_class_ratio=None,
               positive_weight_ratio=4, exclusion_loss_ratio=0,
               imgaug_train_seq=None,
               epoch_traintop=2, epoch_finetuning=15,
               batch_size_train=32, batch_size_valid=32,
               dict_lr_traintop=None, dict_lr_finetuning=None,
               smooth_factor=0,
               model_save_dir='/tmp'
               ):

    #region load csv files

    print('loading csv data')
    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(filename_csv_valid)

    # region get the number of labels in every class
    df = pd.read_csv(filename_csv_train)
    # NUM_CLASSES = df['labels'].nunique(dropna=True) #only work for single label

    LIST_CLASS_SAMPLES_NUM = [0 for _ in range(num_classes)]
    for _, row in df.iterrows():
        labels = str(row["labels"])  # single label may be int type
        list_labels = labels.split('_')
        for label in list_labels:
            if label == '':
                continue

            assert label.isdigit(), 'Error label!'
            LIST_CLASS_SAMPLES_NUM[int(label)] += 1

    # endregion

    # region multiple labels convert to simgle label(smallest class num) in order to dynamic resampling
    train_labels_single = [0 for _ in range(len(train_labels))]

    for i, labels in enumerate(train_labels):
        labels = str(labels)  # single label may be int type
        list_labels = labels.split('_')

        label_current = None
        for label in list_labels:
            if label == '':
                continue

            if label_current is None:  # it is the first label
                label_current = label
            elif LIST_CLASS_SAMPLES_NUM[int(label)] < LIST_CLASS_SAMPLES_NUM[int(label_current)]:
                label_current = label  # current label's class contain smaller number of images than before

        train_labels_single[i] = label_current

    # endregion

    print('loading csv data complete!')

    #endregion

    #region  data generator
    image_shape = (image_size, image_size, 3)
    file_weight_power = os.path.join(sys.path[0], 'weight_power.txt')

    my_gen_train = My_gen_weight_multi_labels(
        files=train_files, labels=train_labels, labels_single=train_labels_single,
        image_shape=image_shape, file_weight_power=file_weight_power,
        list_class_samples_num=LIST_CLASS_SAMPLES_NUM, num_class=num_classes,
        smooth_factor=smooth_factor,
        imgaug_seq=imgaug_train_seq, batch_size=batch_size_train)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels,
                                   image_shape=image_shape, multi_labels=True,
                                   num_output=num_classes, batch_size=batch_size_valid)
    #endregion

    #region custom loss function
    if class_weights is None:
        if inter_class_ratio is not None:
            class_positive_weights = op_class_weight(LIST_CLASS_SAMPLES_NUM, weight_power=inter_class_ratio)
            class_positive_weights = np.array(class_positive_weights)
            # class_samples_weights[0] /= (3)  #class0 Normal
            class_positive_weights *= positive_weight_ratio
        else:
            class_positive_weights = [1 for _ in range(num_classes)]
            class_positive_weights = np.array(class_positive_weights)

        print(np.round(class_positive_weights, 2))

        class_weights = []
        for class_weight1 in class_positive_weights:
            class_weights.append([1, class_weight1])  # sigmoid : 0:1, 1:weight1

        class_weights = np.array(class_weights)

    custom_loss_function = get_weighted_binary_crossentropy(class_weights, exclusion_loss_ratio)

    #endregion

    # region loading and converting model
    if isinstance(model1, str):
        print('loading model...')
        model1 = keras.models.load_model(model1, compile=False)
        print('loading model complete!')

    if add_top:
        model1 = my_transfer_learning.add_top(model1, num_output=num_classes, activation_function='Sigmoid')

    model_train_top = my_transfer_learning.convert_model_transfer(model1,
        change_top=change_top, clsss_num=num_classes, activation_function='Sigmoid',
        freeze_feature_extractor=True, freeze_layes_num=freeze_layes_num)
    model_train_top.summary()

    if GPU_NUM > 1:
        print('convert train header model to Multiple GPU...')
        model_train_top = ModelMGPU(model_train_top, GPU_NUM)
        print('convert train header model to Multiple GPU OK')

    op_adam_train_top = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_train_top.compile(loss=custom_loss_function,
                       optimizer=op_adam_train_top, metrics=['acc'])
    from LIBS.CNN_Models.optimization.lookahead import Lookahead
    lookahead = Lookahead(k=5, alpha=0.5)
    lookahead.inject(model_train_top)
    #endregion

    # region train top layers
    if epoch_traintop > 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        save_filepath_traintop = os.path.join(model_save_dir,
                            model_name + "-traintop-{epoch:03d}-{val_acc:.3f}.hdf5")
        checkpointer_traintop = ModelCheckpoint(save_filepath_traintop, verbose=1,
                            save_weights_only=False, save_best_only=False)

        def scheduler_traintop(epoch):
            if dict_lr_traintop is not None:
                dict_lr_rate = dict_lr_traintop

                for (k, v) in dict_lr_rate.items():
                    if epoch >= int(k):
                        lr_rate = v

                print("epoch：%d, current learn rate:  %f" % (epoch, lr_rate))
                K.set_value(model_train_top.optimizer.lr, lr_rate)

            return K.get_value(model_train_top.optimizer.lr)

        change_lr_traintop = keras.callbacks.LearningRateScheduler(scheduler_traintop)

        history_traintop = model_train_top.fit_generator(
            my_gen_train.gen(),
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train), #number of training batch
            epochs=epoch_traintop,
            validation_data=my_gen_valid.gen(),
            validation_steps=math.ceil(len(valid_files) / batch_size_valid),
            callbacks=[checkpointer_traintop, change_lr_traintop]
        )

    #endregion

    #region fine tuning all layers
    if epoch_finetuning > 0:
        model_fine_tune = my_transfer_learning.convert_trainable_all(model_train_top)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        save_filepath_finetuning = os.path.join(model_save_dir, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")
        checkpointer_finetuning = ModelCheckpoint(save_filepath_finetuning, verbose=1,
                           save_weights_only=False, save_best_only=False)

        def scheduler_fine_tuning(epoch):
            try:
                file_object = open('lr.txt')
                line = file_object.readline()
                file_object.close()
                line = line.strip('\n')  # 删除换行符
                lr_rate = float(line)

                print("epoch：%d, current learn rate by lr.txt:  %f" % (epoch, lr_rate))
                K.set_value(model_fine_tune.optimizer.lr, lr_rate)

            except Exception:
                if dict_lr_finetuning is not None:
                    dict_lr_rate = dict_lr_finetuning

                    for (k, v) in dict_lr_rate.items():
                        if epoch >= int(k):
                            lr_rate = v

                    print("epoch：%d, current learn rate automatically:  %f" % (epoch, lr_rate))
                    K.set_value(model_fine_tune.optimizer.lr, lr_rate)

            return K.get_value(model_fine_tune.optimizer.lr)

        change_lr_finetuning = keras.callbacks.LearningRateScheduler(scheduler_fine_tuning)

        history_fine_tuning = model_fine_tune.fit_generator(
            my_gen_train.gen(),
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train), #number of training batch
            epochs=epoch_finetuning,
            validation_data=my_gen_valid.gen(),
            validation_steps=math.ceil(len(valid_files) / batch_size_valid),
            callbacks=[checkpointer_finetuning, change_lr_finetuning]
        )

    #endregion

    K.clear_session()  #release GPU memory


#region training parameters
NUM_CLASSES = 30
IMAGE_SIZE = 299

filename_csv_train = os.path.join(os.path.abspath('..'),
           'datafiles/2020_3_13', 'DLP_patient_based_split_train.csv')
filename_csv_valid = os.path.join(os.path.abspath('..'),
           'datafiles/2020_3_13', 'DLP_patient_based_split_valid.csv')

#used for customized cost function
# inter class balanced, complementary to weight_power
inter_class_ratio = 0.11
# enlarge POSITIVE_WEIGHT_RATIO will predict too many positive labels
positive_weight_ratio = 2.4

# exclusion_loss_ratio = 0.0006  #0.0001, 0.001, 0.01, 0.1
exclusion_loss_ratio = 0

smooth_factor = 0.1

if exclusion_loss_ratio == 0:
    model_save_dir = '/tmp3/2019_9_9/bigclass_30_param{}_{}/'.format(inter_class_ratio, positive_weight_ratio)
else:
    model_save_dir = '/tmp3/2019_9_9/bigclass_30_param{}_{}_exclu_{}/'.format(inter_class_ratio, positive_weight_ratio, exclusion_loss_ratio)

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.96, aug)
imgaug_train_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images
    iaa.CropAndPad(percent=(-0.03, 0.03)),
    # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
    # iaa.Sometimes(0.9, iaa.ContrastNormalization((0.9, 1.1))),
    # iaa.Sometimes(0.9, iaa.Add((-6, 6))),
    sometimes(iaa.Affine(
        # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
        translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
        # translate by -20 to +20 percent (per axis)
        rotate=(-15, 15),  # rotate by -10 to +10 degrees
    )),
])

dict_lr_traintop = collections.OrderedDict()
dict_lr_traintop['0'] = 3e-4
dict_lr_traintop['1'] = 1e-5
# dict_lr_rate_traintop['2'] = 1e-5

dict_lr_finetuning = collections.OrderedDict()
dict_lr_finetuning['0'] = 1e-5  # 0.00001
dict_lr_finetuning['1'] = 1e-6
# dict_lr_rate_finetuning['4'] = 6e-7

#endregion

#region start training
MODEL_SAVE_DIR = '/media/ubuntu/data1/tmp5/2020_3_13_multabels_two_steps'

for model_name in ['InceptionV3', 'Xception', 'InceptionResnetV2']:
    model_save_dir = os.path.join(MODEL_SAVE_DIR,
            'bigclass_30_param{}_{}/'.format(inter_class_ratio, positive_weight_ratio),  model_name)
    if model_name == 'Xception':
        # model_file1 = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_Xception-015-train0.9671_val0.945.hdf5'
        model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/Transfer_learning/Xception-007-0.954.hdf5'
        train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
                   model_name= model_name, model1=model_file,
                   num_classes=NUM_CLASSES, image_size=IMAGE_SIZE,
                   add_top=False, change_top=True, freeze_layes_num=None,
                   inter_class_ratio=inter_class_ratio, positive_weight_ratio=positive_weight_ratio,
                   exclusion_loss_ratio = exclusion_loss_ratio, smooth_factor=smooth_factor,
                   imgaug_train_seq=imgaug_train_seq,
                   epoch_traintop=3, epoch_finetuning=3,
                   dict_lr_traintop=dict_lr_traintop,
                   dict_lr_finetuning=dict_lr_finetuning,
                   batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_VALID,
                   model_save_dir=model_save_dir)

    if model_name == 'InceptionResnetV2':
        # model_file2 = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_InceptionResNetV2-006-train0.9674_val0.951.hdf5'
        model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/Transfer_learning/InceptionResnetV2-006-0.962.hdf5'
        train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
                   model_name=model_name, model1=model_file,
                   num_classes=NUM_CLASSES, image_size=IMAGE_SIZE,
                   add_top=False, change_top=True, freeze_layes_num=None,
                   inter_class_ratio=inter_class_ratio, positive_weight_ratio=positive_weight_ratio,
                   exclusion_loss_ratio = exclusion_loss_ratio, smooth_factor=smooth_factor,
                   imgaug_train_seq=imgaug_train_seq,
                   epoch_traintop=3, epoch_finetuning=3,
                   dict_lr_traintop=dict_lr_traintop,
                   dict_lr_finetuning=dict_lr_finetuning,
                   batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_VALID,
                   model_save_dir=model_save_dir)

    if model_name == 'InceptionV3':
        model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'
        train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
                   model_name=model_name, model1=model_file,
                   num_classes=NUM_CLASSES, image_size=IMAGE_SIZE,
                   add_top=False, change_top=True, freeze_layes_num=None,
                   inter_class_ratio=inter_class_ratio, positive_weight_ratio=positive_weight_ratio,
                   exclusion_loss_ratio = exclusion_loss_ratio, smooth_factor=smooth_factor,
                   imgaug_train_seq=imgaug_train_seq,
                   epoch_traintop=4, epoch_finetuning=3,
                   dict_lr_traintop=dict_lr_traintop,
                   dict_lr_finetuning=dict_lr_finetuning,
                   batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_VALID,
                   model_save_dir=model_save_dir)

'''

#region efficient net

IMAGE_SIZE = 300
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64


# EfficientNetB0 - (224, 224, 3)
# EfficientNetB1 - (240, 240, 3)
# EfficientNetB2 - (260, 260, 3)
# EfficientNetB3 - (300, 300, 3)
# EfficientNetB4 - (380, 380, 3)
# EfficientNetB5 - (456, 456, 3)
# EfficientNetB6 - (528, 528, 3)
# EfficientNetB7 - (600, 600, 3)


from efficientnet import EfficientNetB0,EfficientNetB2, EfficientNetB3, EfficientNetB5
base_model = EfficientNetB3(weights='imagenet', include_top=False)

dict_lr_rate_traintop = collections.OrderedDict()
dict_lr_rate_traintop['0'] = 1e-3
dict_lr_rate_traintop['1'] = 1e-4

dict_lr_rate_finetuning = collections.OrderedDict()
dict_lr_rate_finetuning['0'] = 1e-3
dict_lr_rate_finetuning['2'] = 3e-4
dict_lr_rate_finetuning['3'] = 1e-4
dict_lr_rate_finetuning['4'] = 3e-5
dict_lr_rate_finetuning['5'] = 1e-5
dict_lr_rate_finetuning['6'] = 3e-6  # 0.000003
dict_lr_rate_finetuning['7'] = 1e-6  # 0.000001
dict_lr_rate_finetuning['10'] = 6e-7

model_name3 = 'EfficientNetB3'
train_task(filename_csv_train=filename_csv_train, filename_csv_valid=filename_csv_valid,
           model_name=model_name3, model1=base_model, image_size=IMAGE_SIZE,
           add_top=True, change_top=False,
           inter_class_ratio=inter_class_ratio, positive_weight_ratio=positive_weight_ratio,
           imgaug_train_seq=imgaug_train_seq,
           epoch_header=2, epoch_finetuning=12,
           dict_lr_rate_traintop=dict_lr_rate_traintop,
           dict_lr_rate_finetuning=dict_lr_rate_finetuning,
           batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_VALID,
            model_save_dir=model_save_dir)

#endregion

'''

# model_name3 = 'InceptionV3'
# model_file3 = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/InceptionV3-020-train0.9533_val0.945.hdf5'

#endregion

print('OK')
