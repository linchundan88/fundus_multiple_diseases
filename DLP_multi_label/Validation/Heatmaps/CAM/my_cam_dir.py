
import os
from LIBS.Heatmaps.my_helper_heatmaps_CAM import get_CNN_model, server_cam, \
    server_grad_cam, server_gradcam_plusplus
import shutil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

DO_PREPROCESS = True
GENERATE_CSV = True


dir_original = '/tmp5/heatmaps_2020_3_21/original'
dir_preprocess = '/tmp5/heatmaps_2020_3_21/preprocess384'
DIR_DEST_BASE = '/tmp5/heatmaps_2020_3_21/results/CAM_blend_original'

if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess, image_size=384,
                   add_black_pixel_ratio=0.07)

prefix = 'screening_singlelabel_2020_3_21'
filename_csv = os.path.join(DIR_DEST_BASE, prefix + '.csv')

blend_original_image = True

from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
if GENERATE_CSV:
    write_csv_dir_nolabel(filename_csv, dir_preprocess)

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


heatmap_type = 'CAM'

import pandas as pd
df = pd.read_csv(filename_csv)

for _, row in df.iterrows():
    image_file = row['images']
    image_label = int(row['labels'])

    preprocess = False
    input_shape = (299, 299, 3)

    img_source = image_file
    file_dir, filename = os.path.split(img_source)
    file_basename, file_ext = os.path.splitext(filename)

    from LIBS.ImgPreprocess.my_image_helper import my_gen_img_tensor
    img_input = my_gen_img_tensor(image_file, image_shape=input_shape)

    for model_no in range(len(dicts_models)):
        model1 = dicts_models[model_no]['model_original']
        probs = model1.predict(img_input)

        for j in range(len(probs[0])):  # number of classes
            if probs[0][j] > 0.5:
                print(j)

                if heatmap_type == 'grad_cam':
                    filename_CAM1 = server_grad_cam(dicts_models=dicts_models, model_no=model_no,
                                                    img_source=img_source, pred=j,
                                                    preprocess=False, blend_original_image=blend_original_image,
                                                    base_dir_save='/tmp/temp_cam/')
                if heatmap_type == 'CAM':
                    filename_CAM1 = server_cam(dicts_models=dicts_models, model_no=model_no,
                                               img_source=img_source, pred=j,
                                               cam_relu=True, preprocess=False, blend_original_image=blend_original_image,
                                               base_dir_save='/tmp/temp_cam/')
                if heatmap_type == 'gradcam_plus':
                    filename_CAM1 = server_gradcam_plusplus(dicts_models=dicts_models, model_no=model_no,
                                                            img_source=img_input, pred=j,
                                                            preprocess=False, blend_original_image=blend_original_image,
                                                            base_dir_save='/tmp3/temp_cam/')

                file_dest = os.path.join(DIR_DEST_BASE, heatmap_type,
                                         file_dir, file_basename +'_model_' + str(model_no) + '_label_' + str(j)
                                         + file_ext)

                file_dest = file_dest.replace(dir_preprocess, DIR_DEST_BASE)

                if not os.path.exists(os.path.dirname(file_dest)):
                    os.makedirs(os.path.dirname(file_dest))

                print(file_dest)
                shutil.copy(filename_CAM1, file_dest)

print('OK')

