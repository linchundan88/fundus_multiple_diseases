'''

'''
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from LIBS.Heatmaps.deepshap.my_helper_deepshap import get_e_list, shap_deep_explain
import pandas as pd
from keras.layers import *
import keras
import sys

sys.path.append(".")
sys.path.append("..")
from LIBS.ImgPreprocess import my_preprocess
import shutil

REFERENCE_FILE = 'ref_rop.npy'
NUM_REFERENCE = 24
DIR_SAVE_TMP = '/tmp5/tmp'
# DIR_SAVE_RESULTS = '/tmp5/DeepShap_stage_Xception/2020_2_22'
# DIR_SAVE_RESULTS = '/tmp5/DeepShap_stage_InceptionResnetV2/2020_2_22'
DIR_SAVE_RESULTS = '/tmp5/DeepShap_stage_InceptionV3/2020_2_28'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset/Stage/preprocess384/'


dicts_models = []
base_model_dir = '/home/ubuntu/dlp/deploy_models_2019/bigclasses_multilabels_new/bigclass_30_param0.11_2.4'
model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-traintop-001-0.919.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model1)
model2 = {'model_file': os.path.join(base_model_dir, 'Xception-traintop-001-0.910.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model2)
model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-traintop-001-0.913.hdf5'),
          'image_size': 299, 'model_weight': 1}
dicts_models.append(model3)

for dict1 in dicts_models:
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)

e_list = get_e_list(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)


#region generate heatmaps

MODEL_NO = 0
image_shape = dicts_models[MODEL_NO]['input_shape']

for predict_type_name in ['Stage_split_patid_train', 'Stage_split_patid_valid', 'Stage_split_patid_test']:
    save_dir = os.path.join(DIR_SAVE_RESULTS, predict_type_name)
    filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",  "..", "..",
                    'datafiles/dataset10', predict_type_name + '.csv'))
    df = pd.read_csv(filename_csv)

    for _, row in df.iterrows():
        image_file = row['images']
        image_label = int(row['labels'])

        # region predict label
        preprocess = False
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                             image_shape=image_shape)
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                             image_shape=image_shape)

        prob = dicts_models[MODEL_NO]['model'].predict(img_input)
        class_predict = np.argmax(prob)
        #endregion

        if (class_predict == 1 and image_label == 1) or\
                (class_predict == 1 and image_label == 0):

            list_classes, list_images = shap_deep_explain(
                dicts_models=dicts_models, model_no=MODEL_NO,
                e_list=e_list, num_reference=NUM_REFERENCE,
                img_source=image_file, preprocess=False, ranked_outputs=1, base_dir_save=DIR_SAVE_TMP)

            if class_predict == 1 and image_label == 1:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '1_1/'))

            if class_predict == 1 and image_label == 0:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '0_1/'))

            if not os.path.exists(os.path.dirname(file_dest)):
                os.makedirs(os.path.dirname(file_dest))

            print(file_dest)
            shutil.copy(list_images[0], file_dest)

#endregion

print('OK')