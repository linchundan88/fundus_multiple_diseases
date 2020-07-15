'''
run mode :5 seconds xception
debug mode :10 seconds
RPC  deep_explain, deep_lift(rescale)

'''

import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#limit gpu memory usage
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from deepexplain.tensorflow import DeepExplain
from xmlrpc.server import SimpleXMLRPCServer
from keras.layers import *
from keras.models import Model
import keras
import sys, time
import uuid
sys.path.append("./")
sys.path.append("../")
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess
from matplotlib import pylab as plt
from multiprocessing import Queue
from multiprocessing import Process


TIME_OUT = 30

NUM_CLASSES = 2

#server_cam 不传递文件名称，自动保存位置
BASE_DIR_SAVE = '/tmp'
if not os.path.exists(BASE_DIR_SAVE):
    os.makedirs(BASE_DIR_SAVE)


q_process_start = Queue()  #indicate loading and converting model OK
q_request = Queue()
q_response = Queue()


#indepedent process
def process_deep_explain(model, num_classes=NUM_CLASSES):

    print('prepare to load model:' + model['model_file'])
    model1 = keras.models.load_model(model['model_file'])
    print('model load complete!')

    image_size = model['image_size']

    with DeepExplain(session=K.get_session()) as de:  # init DeepExplain context

        #region convert model: input_tensor, target_tensor

        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # With Keras this is very easy:
        # 1. Get the input tensor to the original model
        input_tensor = model1.layers[0].input

        # 2. We now target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last dense (index -2)
        fModel = Model(inputs=input_tensor, outputs=model1.layers[-1].output)
        target_tensor = fModel(input_tensor)

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
                img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384, add_black_pixel_ratio=0)
                img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                                      image_shape=(image_size, image_size, 3))
            else:
                img_preprocess = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                                                                      image_shape=(image_size, image_size, 3))
            #endregion

            #region generate deep_lift maps

            xs = img_preprocess
            ys = keras.utils.to_categorical([pred], num_classes)

            # attributions = de.explain('grad*input', target_tensor * ys, input_tensor, xs)
            # attributions = de.explain('saliency', target_tensor * ys, input_tensor, xs)
            # attributions = de.explain('intgrad', target_tensor * ys, input_tensor, xs)
            # 4 seconds
            attributions = de.explain('deeplift', target_tensor * ys, input_tensor, xs)
            # attributions = de.explain('elrp', target_tensor * ys, input_tensor, xs)
            # attributions = de.explain('occlusion', target_tensor * ys, input_tensor, xs)


            data1 = attributions[0]
            data1 = np.mean(data1, 2)

            # data1 = attributions[0][:, :, 0].reshape(image_size, image_size)
            data = data1.reshape(image_size, image_size)

            abs_max = np.percentile(np.abs(data), 100)
            abs_min = abs_max

            cmap = 'seismic'    # cmap = 'RdBu_r' cmap = 'gray'

            plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
            plt.axis('off')

            # endregion

            #region  save file and return
            # 传过来的是web目录
            (filepath, tempfilename) = os.path.split(img_source)
            image_uuid = filepath.split('/')[-1]
            filename_CAM = os.path.join(BASE_DIR_SAVE, image_uuid, 'Deep_lift' + str(pred) + '.jpg')

            # 测试可能不存在，实际web上传存在
            if not os.path.exists(os.path.dirname(filename_CAM)):
                os.makedirs(os.path.dirname(filename_CAM))

            plt.savefig(filename_CAM, bbox_inches='tight', )
            plt.close()

            dict1 = {'str_uuid': str_uuid, 'filename_CAM': filename_CAM,
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

    filename_CAM = dict1['filename_CAM']

    return filename_CAM


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

models = []

if reference_class == '0':
    DIR_MODELS = '/home/ubuntu/dlp/english_dr/DR_2classes/'

    # one service ,one model
    if model_no == 0:
        model = {'model_file': os.path.join(DIR_MODELS,
                'Xception-006-0.980.hdf5'),
                 'image_size': 299}


# mdoels, model_no, NUM_CLASSES
p = Process(target=process_deep_explain, args=(model, NUM_CLASSES))
p.start()

print('waiting for deeplift preprocess')

while q_process_start.empty():
    time.sleep(1)

print('deeplift preprocess OK!')

# region test mode
DEGUB_MODE = True
if DEGUB_MODE:
    img_source = '/media/ubuntu/data1/公开数据集/IDRID/B. Disease Grading/IDRID_results_2019_3_13/1/prob100#IDRiD_056.jpg'

    if os.path.exists(img_source):
        for i in range(50):  # take time longer and longer
            time1 = time.time()
            filename_CAM1 = server_deep_explain(img_source, pred=1, preprocess=True)
            time2 = time.time()
            print(time2-time1)


        print('OK')

        print(filename_CAM1)
#endregion


#region run service

# server = SimpleXMLRPCServer(("localhost", port))
server = SimpleXMLRPCServer(("0.0.0.0", port))
print("Listening on port: ", str(port))
server.register_function(server_deep_explain, "server_deep_explain")
server.serve_forever()

#endregion