
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import *
import sys
import shutil
sys.path.append("./")
sys.path.append("../")
from LIBS.ImgPreprocess import my_preprocess

from LIBS.Heatmaps.my_helper_heatmaps_CAM import get_CNN_model, server_cam, \
    server_grad_cam, server_gradcam_plusplus

#region load and convert models

model_dir = '/tmp3/models_ROP_2019_11_14/Grade/train0/transfer0'
dicts_models = []
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-003-0.976.hdf5'),
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

for dict1 in dicts_models:
    print('prepare to load model:' + dict1['model_file'])
    original_model, output_model, all_amp_layer_weights1 = get_CNN_model(dict1['model_file'])

    if 'input_shape' not in dict1:
        if original_model.input_shape[2] is not None:
            dict1['input_shape'] = original_model.input_shape[1:]
        else:
            dict1['input_shape'] = (299, 299, 3)

    dict1['model_original'] = original_model
    dict1['model_cam'] = output_model
    dict1['all_amp_layer_weights'] = all_amp_layer_weights1

    print('model load complete!')

#endregion

#region generate heatmaps
import pandas as pd

dir_preprocess = '/media/ubuntu/data1/ROP项目/preprocess384/'
dir_dest_heatmap = '/tmp5/ROP_CAM_2019_11_19/'

for heatmap_type in ['CAM', 'grad_cam', 'gradcam_plus']:
    for csv_type in ['Stage_split_patid_train', 'Stage_split_patid_valid', 'Stage_split_patid_test']:

        filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", "..",
                    'datafiles/dataset3', csv_type + '.csv'))
        df = pd.read_csv(filename_csv)

        for _, row in df.iterrows():
            image_file = row['images']
            image_label = int(row['labels'])
            # img_source = image_file.replace('/preprocess384/', '/original/')

            preprocess = False
            input_shape = (299, 299, 3)
            if preprocess:
                img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
                img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                                 image_shape=input_shape)
            else:
                img_source = image_file
                img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                                 image_shape=input_shape)

            model1 = dicts_models[0]['model_original']
            probs = model1.predict(img_input)
            class_predict = np.argmax(probs)

            if (class_predict == 1 and image_label == 1) or (class_predict == 1 and image_label == 0):
                if heatmap_type == 'grad_cam':
                    filename_CAM1 = server_grad_cam(dicts_models=dicts_models, model_no=0,
                            img_source=img_source, pred=class_predict,
                            preprocess=False, blend_original_image=True, base_dir_save='/tmp/temp_cam/')
                if heatmap_type == 'CAM':
                    filename_CAM1 = server_cam(dicts_models=dicts_models, model_no=0,
                            img_source=img_source, pred=class_predict,
                            cam_relu=True, preprocess=False, blend_original_image=True, base_dir_save='/tmp/temp_cam/')
                if heatmap_type == 'gradcam_plus':
                    filename_CAM1 = server_gradcam_plusplus(dicts_models=dicts_models, model_no=0,
                            img_source=img_input, pred=class_predict,
                            preprocess=False, blend_original_image=True,
                            base_dir_save='/tmp3/temp_cam/')

                save_dir = os.path.join(dir_dest_heatmap, heatmap_type, csv_type)

                if class_predict == 1 and image_label == 1:
                    file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '1_1/'))

                if class_predict == 1 and image_label == 0:
                    file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '0_1/'))

                if not os.path.exists(os.path.dirname(file_dest)):
                    os.makedirs(os.path.dirname(file_dest))

                print(file_dest)
                shutil.copy(filename_CAM1, file_dest)

#endregion

print('OK!')