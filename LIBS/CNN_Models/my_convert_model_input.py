'''
    Create 2019_3_18
    modify input dimensions of a saved model
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras import backend as K
import numpy as np
import gc
import traceback

from multiprocessing import pool

def change_model_input(model_org_file, model_save_file,
       new_input_shape = (None, 299, 299, 3)):

    try:
        print('load original model...')
        model = keras.models.load_model(model_org_file, compile=False)
        print('load original model complete')

        if model.input_shape[2] is not None:
            print('do not need process:', model_org_file)
            return

        # replace input shape of first layer
        model._layers[1].batch_input_shape = new_input_shape

        print('rebuild new model...')
        # rebuild model architecture by exporting and importing via json
        new_model = keras.models.model_from_json(model.to_json())
        # new_model.summary()
        print('rebuild new model OK')

        print('save weights ...')
        model.save_weights('/tmp/model.hdf5')
        print('save weights complete')
        print('load weights ...')
        new_model.load_weights('/tmp/model.hdf5')
        print('load weights complete')

        # copy weights from old model to new one
        # for layer in new_model.layers:
        #     try:
        #         layer.set_weights(model.get_layer(name=layer.name).get_weights())
        #     except:
        #         print("Could not transfer weights for layer {}".format(layer.name))


        # test new model on a random input image
        X = np.random.rand(1, 299, 299, 3)
        y_pred1 = model.predict(X)
        y_pred2 = new_model.predict(X)

        if (y_pred1 == y_pred2).all():
            print('save model:', model_save_file, '...')
            new_model.save(model_save_file)
            print('model save complete')

    except Exception:
        print('Error: ', model_org_file)
        print(traceback.format_exc())
        return None

    return new_model



if __name__ == '__main__':
    new_input_shape = (None, 299, 299, 3)
    model_org_file = '/media/ubuntu/data2/tmp/deploy_models_2019/bigclasses_multilabels_new/BigClasses_multi_labels_param_01_3/Xception-011-0.846.hdf5'
    model_save_file = '/tmp/aaa.hdf5'
    new_model = change_model_input(model_org_file, model_save_file, new_input_shape)


    '''
    pool.Pool(processes=10)  # 创建5条进程


    dir = '/media/ubuntu/data2/tmp'

    i = 0

    for dir_path, subpaths, files in os.walk(dir, False):
         for f in files:
             model_file = os.path.join(dir_path, f)
             filename, file_extension = os.path.splitext(model_file)
             if file_extension.upper() not in ['.HDF5', 'H5']:
                 print('other file:', model_file)
                 continue

             change_model(model_file)

    '''

     #         p.apply_async(change_model, (model_file,))
     #         i += 1
     #         if i == 10:
     #             p.close()  # 关闭进程池，不再接受请求
     #             p.join() # 等待所有的子进程结束
     #             K.clear_session()
     #             gc.collect()
     #             i = 0
     #
     # p.close()  # 关闭进程池，不再接受请求
     # p.join()  # 等待所有的子进程结束



