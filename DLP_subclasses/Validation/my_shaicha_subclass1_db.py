import os
import LIBS.DLP.my_dlp_helper
import LIBS.DLP.my_predict_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
GPU_NUM = 1
from LIBS.DataPreprocess import my_data
from LIBS.DataValidation import my_multi_class
from LIBS.DataValidation import my_confusion_matrix
import pandas as pd
import pickle

DO_PREPROCESS = False

GEN_CSV = False
GET_LABELS_FROM_DIR = False

COMPUTE_CONFUSIN_MATRIX = True
COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/tmp2/results_2019_5_3/'

predict_type_name = 'Subclass1'
# filename_csv = os.path.join(DIR_DEST_BASE, predict_type_name + '.csv')
filename_csv = 'DLP_SubClass_1.csv'

dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'

dir_dest_confusion = os.path.join(DIR_DEST_BASE, predict_type_name, 'confusion_matrix', 'files')
dir_dest_predict_dir = os.path.join(DIR_DEST_BASE, predict_type_name, 'dir')
pkl_prob = os.path.join(DIR_DEST_BASE, predict_type_name + '_prob.pkl')
pkl_confusion_matrix = os.path.join(DIR_DEST_BASE, predict_type_name + '_cf.pkl')

subclass_type = '1'

#region define models

# region define models, loading once for train and validation dataset.

if subclass_type == '0.1':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass0.1/'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-023-0.958.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-007-0.955.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '0.2':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass0.2/'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-009-0.848.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-006-0.833.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '1':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass1'
    dicts_models = []

    # dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-006-0.916.hdf5'),
    #                'image_size': 299, 'model_weight': 1}
    # dicts_models.append(dict_model1)
    #
    # dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-010-0.920.hdf5'),
    #                'image_size': 299, 'model_weight': 1}
    # dicts_models.append(dict_model2)

    # model_dir = '/tmp2/models_subclass/DLP_SubClass1'
    # dicts_models = []
    #
    # dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-018-0.916.hdf5'),
    #                'image_size': 299, 'model_weight': 1}
    # dicts_models.append(dict_model1)
    #
    # dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-007-0.918.hdf5'),
    #                'image_size': 299, 'model_weight': 1}
    # dicts_models.append(dict_model2)

    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass1/2019_5_3_add_test_dataset'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-005-0.915.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-011-0.915.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '2':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass2'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-026-0.997.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-036-0.995.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '5':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass5'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-035-0.953.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-031-0.948.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '15':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass15'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-041-0.940.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-060-0.936.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

if subclass_type == '29':
    model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass29'
    dicts_models = []

    dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-019-0.967.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-011-0.966.hdf5'),
                   'image_size': 299, 'model_weight': 1}
    dicts_models.append(dict_model2)

# endregion

#endregion

if DO_PREPROCESS:
    from LIBS.ImgPreprocess import my_preprocess_dir
    image_size = 384
    my_preprocess_dir.do_process_dir(dir_original, dir_preprocess, image_size=image_size)

if GEN_CSV:
    if not os.path.exists(os.path.dirname(filename_csv)):
        os.makedirs(os.path.dirname(filename_csv))

    if GET_LABELS_FROM_DIR:
        dict_mapping = {}
        for i in range(30):
            dict_mapping[str(i)] = str(i)

        my_data.write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping)
    else:
        my_data.write_csv_dir_nolabel(filename_csv, dir_preprocess)


df = pd.read_csv(filename_csv)
all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list =\
    LIBS.DLP.my_predict_helper.do_predict_batch(dicts_models, filename_csv, gpu_num=GPU_NUM)

if not os.path.exists(os.path.dirname(pkl_prob)):
    os.makedirs(os.path.dirname(pkl_prob))
with open(pkl_prob, 'wb') as file:
    pickle.dump(prob_total, file)

# pkl_file = open(prob_pkl', 'rb')
# prob_total = pickle.load(pkl_file)

if COMPUTE_CONFUSIN_MATRIX:
    (cf_list, not_match_list, cf_total, not_match_total) = \
        my_confusion_matrix.compute_confusion_matrix(prob_list, dir_dest_confusion,
             all_files, all_labels, dir_preprocess=dir_preprocess, dir_original=dir_original)

    if not os.path.exists(os.path.dirname(pkl_confusion_matrix)):
        os.makedirs(os.path.dirname(pkl_confusion_matrix))
    with open(pkl_confusion_matrix, 'wb') as file:
        pickle.dump(cf_total, file)

if COMPUTE_DIR_FILES:
    my_multi_class.op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
        dir_dest=dir_dest_predict_dir, dir_original=dir_original, keep_subdir=True)


print('OK')