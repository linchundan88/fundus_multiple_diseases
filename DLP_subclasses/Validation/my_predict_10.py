import os
import LIBS.DLP.my_dlp_helper
import LIBS.DLP.my_predict_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_NUM = 1
from LIBS.DataPreprocess import my_data
from LIBS.DataValidation import my_multi_class
from LIBS.DataValidation import my_confusion_matrix
import pandas as pd

DO_PREPROCESS = False
GEN_CSV = True
GET_LABELS_FROM_DIR = False
COMPUTE_CONFUSIN_MATRIX = False
COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/tmp2/results_2019_4_29/'

# predict_type_name = 'DLP_SubClass10_train'
predict_type_name = 'subclass10_add'
# filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
#                               'datafiles', predict_type_name +'.csv'))
filename_csv = os.path.abspath(os.path.join(predict_type_name +'.csv'))

# dir_original = '/tmp2/SubClass10/original/10/'
# dir_preprocess512 = '/tmp2/SubClass10/preprocess512/10/'
# dir_crop_optic_disc = '/tmp2/SubClass10/Crop_optic_disc_112/10/'

dir_original = '/media/ubuntu/data1/子类/original/10'
dir_preprocess512 = '/media/ubuntu/data1/子类/preprocess512/10'
dir_crop_optic_disc = '/media/ubuntu/data1/子类/Crop_optic_disc_112/10'

dir_dest_confusion = os.path.join(DIR_DEST_BASE, predict_type_name, 'confusion_matrix', 'files')
dir_dest_predict_dir = os.path.join(DIR_DEST_BASE, predict_type_name, 'dir')

pkl_prob = os.path.join(DIR_DEST_BASE, predict_type_name + '_prob.pkl')
pkl_confusion_matrix = os.path.join(DIR_DEST_BASE, predict_type_name + '_cf.pkl')

model_dir = '/home/ubuntu/dlp/deploy_models_2019/subclass_2019_4_10_single_labels/DLP_SubClass10'
dicts_models = []

dict_model1 = {'model_file': os.path.join(model_dir, 'Resnet112-024-0.953.hdf5'),
          'image_size': 112, 'model_weight': 1}
dicts_models.append(dict_model1)

dict_model2 = {'model_file': os.path.join(model_dir, 'Resnet112-036-0.948.hdf5'),
          'image_size': 112, 'model_weight': 1}
dicts_models.append(dict_model2)

if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess512, image_size=512)
    print('preprocess OK!')

    from LIBS.DLP.my_dlp_helper import crop_optic_disc_dir
    crop_optic_disc_dir(dir_source=dir_preprocess512, dir_dest=dir_crop_optic_disc, server_port=21000, mask=True)
    print('crop optic disc 112 OK!')

if GEN_CSV:
    if GET_LABELS_FROM_DIR:
        dict_mapping = {}
        for i in range(30):
            dict_mapping[str(i)] = str(i)

        my_data.write_csv_based_on_dir(filename_csv, dir_crop_optic_disc, dict_mapping)
    else:
        my_data.write_csv_dir_nolabel(filename_csv, dir_crop_optic_disc)

df = pd.read_csv(filename_csv)
all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list =\
    LIBS.DLP.my_predict_helper.do_predict_batch(dicts_models, filename_csv, gpu_num=GPU_NUM)

import pickle
if not os.path.exists(os.path.dirname(pkl_prob)):
    os.makedirs(os.path.dirname(pkl_prob))
with open(pkl_prob, 'wb') as file:
    pickle.dump(prob_total, file)

# pkl_file = open(pkl_prob, 'rb')
# prob_total = pickle.load(pkl_file)

if COMPUTE_CONFUSIN_MATRIX:
    (cf_list, not_match_list, cf_total, not_match_total) = \
    my_confusion_matrix.compute_confusion_matrix(prob_list, dir_dest_confusion,
         all_files, all_labels, dir_preprocess=dir_crop_optic_disc, dir_original=dir_original)

    if not os.path.exists(os.path.dirname(pkl_confusion_matrix)):
        os.makedirs(os.path.dirname(pkl_confusion_matrix))
    with open(pkl_confusion_matrix, 'wb') as file:
        pickle.dump(cf_total, file)

if COMPUTE_DIR_FILES:
    my_multi_class.op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_crop_optic_disc,
        dir_dest=dir_dest_predict_dir, dir_original=dir_original, keep_subdir=True)



print('OK')


'''
confusion

train:
[1102,73]
[46,956]

validation:
[203,12]
[9,161]
'''