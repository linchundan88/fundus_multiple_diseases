import os
import LIBS.DLP.my_dlp_helper
import LIBS.DLP.my_predict_helper

from LIBS.DataPreprocess import my_data
from LIBS.DataValidation import my_multi_class
from LIBS.DataValidation import my_confusion_matrix
import pandas as pd
import pickle

CUDA_VISIBLE_DEVICES = "2"
GPU_NUM = 1

DO_PREPROCESS = True

GEN_CSV = True

COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/tmp5/测试集分子类/results'

dir_original = '/tmp5/测试集分子类/original'
dir_preprocess = '/tmp5/测试集分子类/preprocess384'

if DO_PREPROCESS:
    from LIBS.ImgPreprocess import my_preprocess_dir
    image_size = 384
    my_preprocess_dir.do_process_dir(dir_original, dir_preprocess,
            image_size=image_size, add_black_pixel_ratio=0.02)

    print('Preprocess OK')

for subclass_type in ['0.1', '0.2', '1', '2', '29']:
# for subclass_type in ['0.1', '0.2', '1', '2', '5', '15', '29']:

    dir_original_subclass = os.path.join(dir_original, subclass_type)
    dir_preprocess_subclass = os.path.join(dir_preprocess, subclass_type)

    predict_type_name = 'Subclass' + subclass_type
    filename_csv = os.path.join(DIR_DEST_BASE, predict_type_name + '.csv')

    if GEN_CSV:
        if not os.path.exists(os.path.dirname(filename_csv)):
            os.makedirs(os.path.dirname(filename_csv))

        my_data.write_csv_dir_nolabel(filename_csv, dir_preprocess_subclass)

    dir_dest_confusion = os.path.join(DIR_DEST_BASE, predict_type_name, 'confusion_matrix', 'files')
    dir_dest_predict_dir = os.path.join(DIR_DEST_BASE, predict_type_name, 'dir')
    pkl_prob = os.path.join(DIR_DEST_BASE, predict_type_name + '_prob.pkl')
    pkl_confusion_matrix = os.path.join(DIR_DEST_BASE, predict_type_name + '_cf.pkl')

    #region define models
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
        # model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass0.2/'
        # dicts_models = []
        #
        # dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-009-0.848.hdf5'),
        #                'image_size': 299, 'model_weight': 1}
        # dicts_models.append(dict_model1)
        #
        # dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-006-0.833.hdf5'),
        #                'image_size': 299, 'model_weight': 1}
        # dicts_models.append(dict_model2)

        model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass0.2/'
        dicts_models = []

        dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-013-0.870.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model1)

        dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-019-0.873.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model2)

    if subclass_type == '1':
        model_dir = '/home/ubuntu/dlp/deploy_models_2019/models_subclass_2019_4_26/DLP_SubClass1'
        dicts_models = []

        dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-017-0.928.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model1)

        dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-017-0.930.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model2)

        # model_dir = '/home/ubuntu/dlp/deploy_models_2019/Neovascularization(Deprecated)'
        # dicts_models = []
        #
        # dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-006-0.872.hdf5'),
        #                'image_size': 299, 'model_weight': 1}
        # dicts_models.append(dict_model1)
        #
        # dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-007-0.871.hdf5'),
        #                'image_size': 299, 'model_weight': 1}
        # dicts_models.append(dict_model2)

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

        dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResNetV2-016-0.980.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model1)

        dict_model2 = {'model_file': os.path.join(model_dir, 'Xception-020-0.974.hdf5'),
                       'image_size': 299, 'model_weight': 1}
        dicts_models.append(dict_model2)

    # endregion

    prob_total, y_pred_total, prob_list, pred_list =\
        LIBS.DLP.my_predict_helper.do_predict_batch(dicts_models, filename_csv,
                argmax=True, cuda_visible_devices=CUDA_VISIBLE_DEVICES, gpu_num=GPU_NUM)

    if not os.path.exists(os.path.dirname(pkl_prob)):
        os.makedirs(os.path.dirname(pkl_prob))
    with open(pkl_prob, 'wb') as file:
        pickle.dump(prob_total, file)

    # pkl_file = open(prob_pkl', 'rb')
    # prob_total = pickle.load(pkl_file)


    if COMPUTE_DIR_FILES:
        my_multi_class.op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess_subclass,
                                           dir_dest=dir_dest_predict_dir, dir_original=dir_original_subclass, keep_subdir=True)

    print('subclass {} compute complete!'.format(subclass_type))
    # import keras.backend as K
    # K.clear_session()  # release GPU memory

print('OK')