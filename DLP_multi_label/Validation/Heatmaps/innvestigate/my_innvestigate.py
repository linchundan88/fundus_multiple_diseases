'''https://github.com/albermax/innvestigate'''

import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess
import shutil
import innvestigate.utils
import keras

if __name__ == "__main__":
    dir_preprocess = '/media/ubuntu/data1/ROP项目/preprocess384/'
    dir_dest_heatmap = '/tmp5/ROP_CAM_2019_11_19/'

    model_file = '/tmp3/models_ROP_2019_11_14/Grade/train0/transfer0/InceptionV3-003-0.976.hdf5'
    model1 = keras.models.load_model(model_file, compile=False)

    for i, layer in enumerate(model1.layers):
        layer.name = 'layer' + str(i)

    model = innvestigate.utils.model_wo_softmax(model1)

    for heatmap_type in ['guided_backprop', 'gradient', 'lrp.z', 'lrp.epsilon', 'integrated_gradients']:
        # Create analyzer
        '''['input', 'random', 'gradient', 'gradient.baseline', 
        'input_t_gradient', 'deconvnet', 'guided_backprop', 
        'integrated_gradients', 'smoothgrad', 'lrp', 'lrp.z', 'lrp.z_IB', 
        'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 'lrp.flat', 
        'lrp.alpha_beta', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB',
         'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 
         'lrp.z_plus_fast', 'lrp.sequential_preset_a', 
         'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 
        'lrp.sequential_preset_b_flat', 'deep_taylor', 'deep_taylor.bounded',
         'deep_lift.wrapper', 'pattern.net', 'pattern.attribution']
        '''

        if heatmap_type == 'guided_backprop':
            analyzer = innvestigate.create_analyzer('guided_backprop', model)
        if heatmap_type == 'gradient':
            analyzer = innvestigate.create_analyzer('gradient', model)
        if heatmap_type == 'lrp.z':
            analyzer = innvestigate.create_analyzer('lrp.z', model)
        if heatmap_type == 'lrp.epsilon':
            analyzer = innvestigate.create_analyzer('lrp.epsilon', model)
        if heatmap_type == 'integrated_gradients':
            analyzer = innvestigate.create_analyzer('integrated_gradients', model)
        # analyzer = innvestigate.create_analyzer("deep_taylor", model)
        # analyzer = innvestigate.create_analyzer("lrp.z", model)
        # analyzer = innvestigate.create_analyzer("deep_lift.wrapper", model)

        for csv_type in ['Stage_split_patid_train', 'Stage_split_patid_valid',
                         'Stage_split_patid_test']:
            filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", "..",
                               'datafiles/dataset3', csv_type + '.csv'))
            df = pd.read_csv(filename_csv)

            for _, row in df.iterrows():
                image_file = row['images']
                image_label = int(row['labels'])

                preprocess = False
                image_size = 299
                if preprocess:
                    img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
                    img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                                     image_shape=(image_size, image_size, 3))
                else:
                    img_source = image_file
                    img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                                     image_shape=(image_size, image_size, 3))

                probs = model1.predict(img_input)
                class_predict = np.argmax(probs)

                save_dir = os.path.join(dir_dest_heatmap, heatmap_type, csv_type)

                if (class_predict == 1 and image_label == 1) or (class_predict == 1 and image_label == 0):
                    if class_predict == 1 and image_label == 1:
                        file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '1_1/'))

                    if class_predict == 1 and image_label == 0:
                        file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '0_1/'))

                    if not os.path.exists(os.path.dirname(file_dest)):
                        os.makedirs(os.path.dirname(file_dest))


                    # Apply analyzer w.r.t. maximum activated output-neuron
                    data_heatmap = analyzer.analyze(img_input)

                    # Aggregate along color channels and normalize to [-1, 1]
                    data_heatmap = data_heatmap.sum(axis=np.argmax(np.asarray(data_heatmap.shape) == 3))
                    data_heatmap /= np.max(np.abs(data_heatmap))

                    import uuid
                    str_uuid = str(uuid.uuid1())
                    filename_heatmap= os.path.join('/tmp', str_uuid + '.jpg')

                    plt.imshow(data_heatmap[0], cmap="seismic", clim=(-1, 1))
                    plt.axis('off')
                    plt.savefig(filename_heatmap, dpi=image_size, bbox_inches='tight')

                    # data_heatmap = np.squeeze(data_heatmap)
                    # sizes = np.shape(data_heatmap)
                    # fig = plt.figure()
                    # fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
                    # ax = plt.Axes(fig, [0., 0., 1., 1.])
                    # ax.set_axis_off()
                    # fig.add_axes(ax)
                    # ax.imshow(data_heatmap, cmap="seismic", clim=(-1, 1))
                    # plt.savefig("aaaa.png", dpi = 300)

                    plt.close()

                    if not os.path.exists(os.path.dirname(file_dest)):
                        os.makedirs(os.path.dirname(file_dest))

                    print(file_dest)
                    shutil.copy(filename_heatmap, file_dest)

print('ok')



