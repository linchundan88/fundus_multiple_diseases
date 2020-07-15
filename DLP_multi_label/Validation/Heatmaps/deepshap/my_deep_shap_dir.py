'''

'''
import os
import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from LIBS.Heatmaps.deepshap.my_helper_deepshap import get_e_list, shap_deep_explain
import keras
import sys

sys.path.append("./")
sys.path.append("../")
from LIBS.ImgPreprocess import my_preprocess
import shutil

DO_PREPROCESS = False

REFERENCE_FILE = 'ref.npy'
NUM_REFERENCE = 24

DIR_SAVE_TMP = '/tmp/deepshap'

dir_original = '/tmp5/heatmaps_2020_3_21/original'
dir_preprocess = '/tmp5/heatmaps_2020_3_21/preprocess384'
DIR_DEST_BASE = '/tmp5/heatmaps_2020_3_21/results/DeepShap'


if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess, image_size=384,
                   add_black_pixel_ratio=0.07)

dicts_models = []
base_model_dir = '/home/ubuntu/dlp/deploy_models_2019/bigclasses_multilabels_new/bigclass_30_param0.11_2.4'
model1 = {'model_file': os.path.join(base_model_dir, 'InceptionResnetV2-traintop-001-0.919.hdf5'),
          'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(model1)
model2 = {'model_file': os.path.join(base_model_dir, 'Xception-traintop-001-0.910.hdf5'),
          'input_shape': (299, 299, 3), 'batch_size': 8}
dicts_models.append(model2)
model3 = {'model_file': os.path.join(base_model_dir, 'InceptionV3-traintop-001-0.913.hdf5'),
          'input_shape': (299, 299, 3), 'batch_size': 24}
dicts_models.append(model3)


for dict1 in dicts_models:
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)

e_list = get_e_list(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)


#region generate heatmaps

image_shape = dicts_models[0]['input_shape']

for dir_path, subpaths, files in os.walk(dir_preprocess, False):
    for f in files:
        image_file_source = os.path.join(dir_path, f)
        file_dir, filename = os.path.split(image_file_source)
        file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue

        preprocess = False
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(image_file_source, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                            image_shape=image_shape)
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file_source,
                                            image_shape=image_shape)

        for model_no in range(len(dicts_models)):

            probs = dicts_models[model_no]['model'].predict(img_input)

            list_positive_classes = []
            for j in range(len(probs[0])):  #number of classes
                if probs[0][j] > 0.5:
                    list_positive_classes.append(j)

            if len(list_positive_classes) == 0:
                continue

            list_positive_classes, list_images = shap_deep_explain(
                    dicts_models=dicts_models, model_no=model_no,
                    e_list=e_list, num_reference=NUM_REFERENCE,
                    img_source=image_file_source, preprocess=False, ranked_outputs=len(list_positive_classes),
                        base_dir_save=DIR_SAVE_TMP)

            for k, positive_class1 in enumerate(list_positive_classes):
                file_dest = os.path.join(file_dir.replace(dir_preprocess, DIR_DEST_BASE),
                        file_base + '__model_{0}_label{1}'.format(model_no, positive_class1) + file_ext)

                if not os.path.exists(os.path.dirname(file_dest)):
                    os.makedirs(os.path.dirname(file_dest))
                print(file_dest)
                shutil.copy(list_images[k], file_dest)

#endregion

print('OK')