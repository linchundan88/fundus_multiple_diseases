'''
provide service for both deep_explain and shap_deep_Explainer

run mode :0.5 seconds xception
RPC  deep_explain, deep_lift(rescale)

'''

import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#region command parameters: class type,gpu_no port no
# python RPC_server_deep_shap.py 0_10 2 25000
if len(sys.argv) != 4:  # sys.argv[0]  exe file itself
    reference_class = '0_10'  # bigclass multi_label
    gpu_no = 0
    port = 25000
else:
    reference_class = str(sys.argv[1])
    gpu_no = str(sys.argv[2])
    port = int(sys.argv[3])

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

#endregion



#limit gpu memory usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from xmlrpc.server import SimpleXMLRPCServer
from keras.layers import *
import keras
import time
import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
from LIBS.Heatmaps.deepshap.my_helper_deepshap import get_e_list, shap_deep_explain
import my_config
DIR_MODELS = my_config.dir_deploy_models

REFERENCE_FILE = 'ref.npy'
NUM_REFERENCE = 24  # shap_deep_explain background  24

#server_cam 不传递文件名称，自动保存位置
dir_tmp = os.path.join(my_config.dir_heatmap, 'deepShap')


# used for RPC service
def server_shap_deep_explain(model_no, img_source, preprocess=True,
                     ranked_outputs=1, blend_original_image=False):

    image_shape = dicts_models[model_no]['input_shape']

    if isinstance(img_source, str):
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(
                img_preprocess, image_shape=image_shape)
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(
                img_source, image_shape=image_shape)
    else:
        img_input = img_source

    list_classes, list_images = shap_deep_explain(
        dicts_models=dicts_models, model_no=model_no,
        e_list=e_list, num_reference=NUM_REFERENCE,
        img_source=img_input, preprocess=False, ranked_outputs=ranked_outputs,
        blend_original_image=blend_original_image, base_dir_save=dir_tmp)

    return list_classes, list_images


#region load models
dicts_models = []

if reference_class == '0':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 8}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 8}
    dicts_models.append(dict_model2)

    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 24}
    dicts_models.append(dict_model3)

if reference_class == '0_10':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionResnetV2-traintop-001-0.919.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 8}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/Xception-traintop-001-0.910.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 8}
    dicts_models.append(dict_model2)
    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclasses_multilabels_new/bigclass_30_param0.11_2.4/InceptionV3-traintop-001-0.913.hdf5'),
                   'input_shape': (299, 299, 3), 'batch_size': 24}

for i, dict1 in enumerate(dicts_models):
    print('loading model:{0} '.format(i))
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)
    print('load model:{0} complete!'.format(i))

e_list = get_e_list(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)

#endregion

# region test mode
if my_config.debug_mode:
    img_source = '/tmp1/brvo.jpg'

    if os.path.exists(img_source):
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
        img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                        image_shape=(299, 299, 3))

        prob = dicts_models[0]['model'].predict(img_preprocess)
        pred = np.argmax(prob)
        print(pred)

        for i in range(2):
            print(time.time())
            list_classes, list_images = server_shap_deep_explain(model_no=0,
                  img_source=img_source, preprocess=True, ranked_outputs=1)
            print(time.time())
            print(list_images)


#endregion


#region run service

# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_shap_deep_explain, "server_shap_deep_explain")
server.serve_forever()

#endregion