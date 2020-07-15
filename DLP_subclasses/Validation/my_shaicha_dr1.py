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
GET_LABELS_FROM_DIR = False

COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/tmp5/测试集分子类/results_2020_3_15_a/'

dir_original = os.path.join('/tmp5/测试集分子类/original/0.3_plan_B')
dir_preprocess = os.path.join('/tmp5/测试集分子类/preprocess512/0.3_plan_B')

if DO_PREPROCESS:
    from LIBS.ImgPreprocess import my_preprocess_dir
    image_size = 512
    my_preprocess_dir.do_process_dir(dir_original, dir_preprocess, image_size=image_size, add_black_pixel_ratio=0.02)

predict_type_name = 'Subclass0.3'
filename_csv = os.path.join(DIR_DEST_BASE, predict_type_name + '.csv')

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

dir_dest_confusion = os.path.join(DIR_DEST_BASE, predict_type_name, 'confusion_matrix', 'files')
dir_dest_predict_dir = os.path.join(DIR_DEST_BASE, predict_type_name, 'dir')
pkl_prob = os.path.join(DIR_DEST_BASE, predict_type_name + '_prob.pkl')
pkl_confusion_matrix = os.path.join(DIR_DEST_BASE, predict_type_name + '_cf.pkl')

model_dir = '/home/ubuntu/dlp/deploy_models_2019'
dicts_models = []

dict_model1 = {'model_file': os.path.join(model_dir,'SubClass0_3/ResNet448-005-0.841.hdf5'),
               'image_size': 448, 'model_weight': 1}
dicts_models.append(dict_model1)

df = pd.read_csv(filename_csv)
all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=df)

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
    my_multi_class.op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
        dir_dest=dir_dest_predict_dir, dir_original=dir_original, keep_subdir=True)


print('OK')