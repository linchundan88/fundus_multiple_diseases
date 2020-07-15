'''
run mode :0.5 seconds xception
RPC  deep_explain, deep_lift(rescale)

one RPC Service only sopport one model
'''

import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#limit gpu memory usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from xmlrpc.server import SimpleXMLRPCServer
from keras.layers import *
from keras.models import Model
import keras
import time
import uuid
import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
from matplotlib import pylab as plt
from multiprocessing import Queue
from multiprocessing import Process
from deepexplain.tensorflow import DeepExplain

import my_config
DIR_MODELS = my_config.dir_deploy_models

TIME_OUT = 10

#server_cam 不传递文件名称，自动保存位置
BASE_DIR_SAVE = os.path.join(my_config.dir_heatmap, 'deepExplain')
if not os.path.exists(BASE_DIR_SAVE):
    os.makedirs(BASE_DIR_SAVE)

def plot_heatmap(attributions, save_filename):
    while not isinstance(attributions, np.ndarray):
        attributions = attributions[0]

    # attributions.shape: (1, 299, 299, 3)
    data = attributions[0]
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
    # plt.axis('off')
    plt.savefig(save_filename, bbox_inches='tight', )

    plt.close()

#indepedent process
def process_deep_explain(dict1):

    with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context

        #region loading and converting model, init get_explainer
        print('loading model...')
        model1 = keras.models.load_model(dict1['model_file'], compile=False)
        print('loading model complete')

        if 'image_size' in dict1:
            image_size = dict1['image_size']
        elif model1.input_shape[2] is not None:
            image_size = model1.input_shape[2]
        else:
            image_size = 299

        NUM_CLASSES = model1.output_shape[1]

        print('converting model...')
        # otherwise ,de.explain( error
        img_preprocess = np.zeros((1, image_size, image_size, 3))
        prob = model1.predict(img_preprocess)

        input_tensor = model1.layers[0].input
        # input
        fModel = Model(inputs=input_tensor, outputs=model1.layers[-1].output)
        target_tensor = fModel(input_tensor)

        # if model.layers[0].input (None, None, None, 3) (299,299,3)
        # explainer = de.get_explainer('deeplift', target_tensor, input_tensor)
        baseline = np.zeros((image_size, image_size, 3))
        explainer = de.get_explainer('deeplift', target_tensor, input_tensor,
                                     baseline=baseline)

        print('converting model complete')
        q_process_start.put('OK')

        #endregion

        while (True):
            while q_request.empty():
                time.sleep(0.1)

            #region get parameters from request queue  and preprocess if needed

            dict_req_param = q_request.get()

            date_time_start = dict_req_param['date_time_start']
            if time.time() - date_time_start > TIME_OUT:
                continue

            str_uuid = dict_req_param['str_uuid']

            img_source = dict_req_param['img_source']
            pred = dict_req_param['pred']
            preprocess = dict_req_param['preprocess']

            if preprocess:
                img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
                img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                                      image_shape=(image_size, image_size, 3))
            else:
                img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                                      image_shape=(image_size, image_size, 3))
            #endregion

            #region generate deep_lift maps

            xs = img_preprocess
            ys = keras.utils.to_categorical([pred], NUM_CLASSES)

            attributions = explainer.run(xs, ys=ys, batch_size=1)

            # endregion

            #region  save file and return
            filename_deep_explain = os.path.join(BASE_DIR_SAVE, str_uuid,
                                 'Deep_lift{}.jpg'.format(pred))

            # 测试可能不存在，实际web上传存在
            if not os.path.exists(os.path.dirname(filename_deep_explain)):
                os.makedirs(os.path.dirname(filename_deep_explain))

            plot_heatmap(attributions, filename_deep_explain)

            dict1 = {'str_uuid': str_uuid, 'filename_deep_explain': filename_deep_explain,
                 'date_time_start':date_time_start, 'date_time_end': time.time()}

            q_response.put(dict1)

            #endregion


# RPC Service, put request to queue
def server_deep_explain(filename, pred, preprocess=True):
    str_uuid = str(uuid.uuid1())
    date_time_start = time.time()

    dict1 = {'date_time_start': date_time_start, 'str_uuid': str_uuid,
             'img_source': filename, 'pred': pred, 'preprocess': preprocess}

    q_request.put(dict1)

    while True:
        time_current = time.time()
        if time_current - date_time_start > TIME_OUT:
            print(date_time_start)
            print(time_current)
            print('timeout')
            return None

        if q_response.empty():
            time.sleep(0.1)
        else:
            dict1 = q_response.get()

            str_uuid_return = dict1['str_uuid']

            if str_uuid_return == str_uuid:
                break

    filename_deep_explain = dict1['filename_deep_explain']

    return filename_deep_explain


#region command parameters: class type,and port no
if len(sys.argv) != 4:  # sys.argv[0]  exe file itself
    reference_class = '0'  # bigclass
    model_no = 0
    port = 24000

else:
    reference_class = str(sys.argv[1])
    model_no = int(sys.argv[2])
    port = int(sys.argv[3])

#endregion

if reference_class == '0':

    # one service ,one model
    # if model_no == 0:
    #     dict1 = {
    #         'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_12/InceptionResNetV2-008-0.947.hdf5'),
    #         'image_size': 299, 'model_weight': 1}
    #
    # if model_no == 1:
    #     dict1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_12/Xception-008-0.951.hdf5'),
    #                'image_size': 299, 'model_weight': 1}

    if model_no == 0:
        dict1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'),
                   'image_size': 299}

    if model_no == 1:
        dict1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'),
                   'image_size': 299}

    if model_no == 2:
        dict1 = {'model_file': os.path.join(DIR_MODELS, 'bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'),
                   'image_size': 299}

q_process_start = Queue()  #indicate loading and converting model OK
q_request = Queue()
q_response = Queue()

p = Process(target=process_deep_explain, args=(dict1,))
p.start()

print('waiting for deeplift preprocess')

while q_process_start.empty():
    time.sleep(1)

print('deeplift preprocess OK!')

# region test mode
if my_config.debug_mode:
    img_source = '/tmp1/img4.jpg'

    if os.path.exists(img_source):
        print(time.time())
        filename_deep_explain = server_deep_explain(img_source, pred=1, preprocess=True)
        print(time.time())

        print(filename_deep_explain)

#endregion


#region run service

# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_deep_explain, "server_deep_explain")
server.serve_forever()

#endregion