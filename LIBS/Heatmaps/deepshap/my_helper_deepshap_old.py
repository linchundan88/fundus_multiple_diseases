'''
provide service for both deep_explain and shap_deep_Explainer

run mode :0.5 seconds xception
RPC  deep_explain, deep_lift(rescale)

'''

import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#limit gpu memory usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from xmlrpc.server import SimpleXMLRPCServer
from keras.layers import *
import keras
import sys, time
import uuid
sys.path.append("./")
sys.path.append("../")
import copy
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess
from matplotlib import pylab as plt
import shap

import my_config
DIR_MODELS = my_config.dir_deploy_models

NUM_REFERENCE = 12  # shap_deep_explain background  12

#server_cam 不传递文件名称，自动保存位置
BASE_DIR_SAVE = os.path.join(my_config.dir_heatmap, 'deepShap')
if not os.path.exists(BASE_DIR_SAVE):
    os.makedirs(BASE_DIR_SAVE)

#region shap_deep_explain
def plot_heatmap_shap(attributions, list_images):

    pred_class_num = len(attributions[0])

    for i in range(pred_class_num):
        # predict_max_class = attributions[1][0][i]
        attribution1 = attributions[0][i]

        #attributions.shape: (1, 299, 299, 3)
        data = attribution1[0]
        data = np.mean(data, -1)

        abs_max = np.percentile(np.abs(data), 100)
        abs_min = abs_max

        # dx, dy = 0.05, 0.05
        # xx = np.arange(0.0, data1.shape[1], dx)
        # yy = np.arange(0.0, data1.shape[0], dy)
        # xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        # extent = xmin, xmax, ymin, ymax

        # cmap = 'RdBu_r'
        # cmap = 'gray'
        cmap = 'seismic'

        # plt.imshow(data1, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        plt.axis('off')

        save_filename1 =list_images[i]
        plt.savefig(save_filename1, bbox_inches='tight', )

        plt.close()

# do not need multi_process, only one method
def server_shap_deep_explain(dicts_models, model_no, img_source, preprocess=True,
                     ranked_outputs=1):

    str_uuid = str(uuid.uuid1())

    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384, tadd_black_pixel_ratio=0)
        img_preprocess = my_images_generator.my_Generator_test_onetime(img_preprocess,
                                           image_shape=(image_size, image_size, 3))
    else:
        img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                              image_shape=(image_size, image_size, 3))

    #mini-batch
    shap_values1 = e_list1[model_no].shap_values(img_preprocess, ranked_outputs=ranked_outputs)
    shap_values2 = e_list2[model_no].shap_values(img_preprocess, ranked_outputs=ranked_outputs)

    shap_values = copy.deepcopy(shap_values1)
    for i in range(ranked_outputs):
        shap_values[0][i] = (shap_values1[0][i] + shap_values2[0][i])/2


    list_classes = []
    list_images = []

    for i in range(ranked_outputs):
        #numpy int 64 - int
        list_classes.append(int(shap_values[1][0][i]))

        save_filename = os.path.join(BASE_DIR_SAVE, str_uuid,
             'Shap_Deep_Explain{}.jpg'.format(shap_values[1][0][i]))

        if not os.path.exists(os.path.dirname(save_filename)):
            os.makedirs(os.path.dirname(save_filename))

        list_images.append(save_filename)


    plot_heatmap_shap(shap_values, list_images)

    return list_classes, list_images

#endregion


#region command parameters: class type,and port no
if len(sys.argv) != 3:  # sys.argv[0]  exe file itself
    reference_class = '0'  # bigclass
    port = 25000

else:
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])

#endregion

#region load models
dicts_models = []

if reference_class == '0':
    dict_model1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model1)

    dict_model2 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model2)

    dict_model3 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'),
                   'image_size': 299}
    dicts_models.append(dict_model3)

for dict1 in dicts_models:
    dict1['model'] = keras.models.load_model(dict1['model_file'], compile=False)

#endregion


image_size = 299

background = np.load('ref.npy')
background1 = background[0:NUM_REFERENCE, :, :, :]
background2 = background[NUM_REFERENCE:NUM_REFERENCE*2, :, :, :]

# x_train = np.zeros((299, 299, 3))
# x_train /= 255.
# x_train -= 0.5
# x_train *= 2.
# background = np.expand_dims(x_train, axis=0)

e_list1 = []
for i in range(len(dicts_models)):
    print('converting model1 ...')
    e = shap.DeepExplainer(dicts_models[0]['model'], background1)  #it will take 10 seconds
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model1.layers[0].input, model1.layers[-1].output), background)
    print('converting model1 complete')
    e_list1.append(e)
e_list2 = []
for i in range(len(dicts_models)):
    print('converting model2 ...')
    e = shap.DeepExplainer(dicts_models[0]['model'], background2)  #it will take 10 seconds
    print('converting model2 complete')
    e_list2.append(e)

# region test mode
DEGUB_MODE = True
if DEGUB_MODE:
    img_source = '/tmp1/img4.jpg'

    if os.path.exists(img_source):
        img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384, train_or_valid='valid')
        img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                              image_shape=(image_size, image_size, 3))

        prob = dicts_models[0]['model'].predict(img_preprocess)
        pred = np.argmax(prob)
        print(pred)

        print('model1')
        #first time take longer
        for i in range(2):
            print(time.time())
            list_classes, list_images = server_shap_deep_explain(dicts_models=dicts_models, model_no=0,
                  img_source=img_source, preprocess=True, ranked_outputs=1)
            print(time.time())
            print(list_images)

        print('model2')
        for i in range(2):
            print(time.time())
            list_classes, list_images = server_shap_deep_explain(dicts_models=dicts_models, model_no=1,
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