'''https://github.com/albermax/innvestigate'''

import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import numpy as np
import innvestigate.utils
import keras

if __name__ == "__main__":

    model_file = '/tmp3/models_ROP_2019_11_14/Grade/train0/transfer0/InceptionV3-003-0.976.hdf5'
    model1 = keras.models.load_model(model_file, compile=False)

    image_file = '/media/ubuntu/data1/ROP项目/preprocess384/ROP训练图集汇总_20190928/本院201808-201908/分期病变/201901190009_OD201901190013_左眼_20190119100434986.jpg'

    from LIBS.DataPreprocess import my_images_generator
    from LIBS.ImgPreprocess import my_preprocess
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

    for i, layer in enumerate(model1.layers):
        layer.name = 'layer' + str(i)

    model = innvestigate.utils.model_wo_softmax(model1)

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

    # analyzer = innvestigate.create_analyzer("deep_taylor", model)
    analyzer = innvestigate.create_analyzer("gradient", model)
    # analyzer = innvestigate.create_analyzer("guided_backprop", model)
    # analyzer = innvestigate.create_analyzer("lrp.z", model)
    # analyzer = innvestigate.create_analyzer("deep_lift.wrapper", model)

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(img_input)
    # a = analyzer.analyze(img_input)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))

    # Plot
    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    plt.savefig("bbb.png", dpi = 300, bbox_inches='tight')

    a = np.squeeze(a)
    sizes = np.shape(a)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(a,  cmap="seismic", clim=(-1, 1))
    plt.savefig("aaaa.png", dpi = 300)


    plt.close()

    print('ok')



