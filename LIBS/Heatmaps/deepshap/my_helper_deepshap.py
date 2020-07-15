import copy
import os
import uuid

import shap
from matplotlib import pylab as plt

import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
import numpy as np

GIF_FPS = 1

def plot_heatmap_shap(attributions,  list_images, img_input, blend_original_image):

    pred_class_num = len(attributions[0])

    if blend_original_image:
        from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse
        img_original = np.uint8(input_norm_reverse(img_input[0]))
        import cv2
        img_original = cv2.resize(img_original, (384, 384))
        img_original_file = os.path.join(os.path.dirname(list_images[0]), 'deepshap_original.jpg')
        cv2.imwrite(img_original_file, img_original)

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
        plt.axis('off')
        # plt.imshow(data1, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        # plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)

        # fig = plt.gcf()
        # fig.set_size_inches(2.99 / 3, 2.99 / 3)  # dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        if blend_original_image:
            # cv2.imwrite('/tmp5/tmp/cv2.jpg', np.uint8(img_input[0]))
            # img_original = cv2.cvtColor(np.uint8(img_input[0]), cv2.COLOR_BGR2RGB)
            # plt.imshow(img_original)

            plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
            save_filename1 = list_images[i]
            plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
            plt.close()

            img_heatmap = cv2.imread(list_images[i])
            img_heatmap = cv2.resize(img_heatmap, (384, 384))
            img_heatmap_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.jpg'.format(i))
            cv2.imwrite(img_heatmap_file, img_heatmap)

            dst = cv2.addWeighted(img_original, 0.65, img_heatmap, 0.35, 0)
            # cv2.imwrite('/tmp5/tmp/aaaaa.jpg', dst) #test code
            img_blend_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_blend_{0}.jpg'.format(i))
            cv2.imwrite(img_blend_file, dst)

            # fig.savefig('/tmp5/tmp/aaa1.png', format='png', dpi=299,  transparent=True,  pad_inches=0)
            # plt.savefig('/tmp5/tmp/aaa.jpg', bbox_inches='tight', pad_inches=0)

            #region create gif
            import imageio
            mg_paths = [img_original_file, img_heatmap_file, img_blend_file]
            gif_images = []
            for path in mg_paths:
                gif_images.append(imageio.imread(path))
            img_file_gif = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.gif'.format(i))
            imageio.mimsave(img_file_gif, gif_images, fps=GIF_FPS)
            list_images[i] = img_file_gif
            #endregion
        else:
            plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
            save_filename1 = list_images[i]
            plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
            plt.close()


def shap_deep_explain(dicts_models, model_no,
                      e_list, num_reference, img_source, preprocess=True,
                      blend_original_image=False,
                      ranked_outputs=1, base_dir_save='/tmp/DeepExplain'):

    image_shape = dicts_models[model_no]['input_shape']

    if isinstance(img_source, str):
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(img_source, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                    image_shape=image_shape)
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_source,
                                    image_shape=image_shape)
    else:
        img_input = img_source

    #region mini-batch because of GPU memory limitation
    list_shap_values = []

    batch_size = dicts_models[model_no]['batch_size']
    split_times = num_reference // batch_size
    for i in range(split_times):
        shap_values_tmp1 = e_list[model_no][i].shap_values(img_input, ranked_outputs=ranked_outputs)
        # shap_values ranked_outputs
        # [0] [0] (1,299,299,3)
        # [1] predict_class array
        shap_values_copy = copy.deepcopy(shap_values_tmp1)
        list_shap_values.append(shap_values_copy)

    for i in range(ranked_outputs):
        for j in range(len(list_shap_values)):
            if j == 0:
                shap_values_tmp2 = list_shap_values[0][0][i]
            else:
                shap_values_tmp2 += list_shap_values[j][0][i]

        shap_values_results = copy.deepcopy(list_shap_values[0])
        shap_values_results[0][i] = shap_values_tmp2 / split_times

    #endregion

    #region save files
    str_uuid = str(uuid.uuid1())
    list_classes = []
    list_images = []
    for i in range(ranked_outputs):
        predict_class = int(shap_values_results[1][0][i]) #numpy int 64 - int
        list_classes.append(predict_class)

        save_filename = os.path.join(base_dir_save, str_uuid,
             'Shap_Deep_Explain{}.jpg'.format(predict_class))
        if not os.path.exists(os.path.dirname(save_filename)):
            os.makedirs(os.path.dirname(save_filename))
        list_images.append(save_filename)

    plot_heatmap_shap(shap_values_results,list_images,
                      img_input, blend_original_image=blend_original_image)

    # because the original image and deepshap image were not aligned.
    # blend_original_image=False,
    # # region blend original
    # if blend_original_image:
    #     for image1 in list_images:
    #         import cv2
    #
    #         if isinstance(img_source, str):
    #             image_original = cv2.imread(img_source)
    #         else:
    #             image_original = img_source
    #
    #         img_deepshap = cv2.imread(image1)
    #         img_deepshap = cv2.resize(img_deepshap, (384, 384))
    #         dst = cv2.addWeighted(img_deepshap, 0.3, image_original, 0.7, 0)
    #
    #         # cv2.imwrite('/tmp5/111.jpg', image_original)
    #         # cv2.imwrite('/tmp5/aaa.jpg', dst)
    #         cv2.imwrite(image1, dst)
    # #endregion

    #endregion

    return list_classes, list_images


def get_e_list(dicts_models, reference_file, num_reference):

    background = np.load(reference_file)

    e_list = []  #model no , background split
    for i in range(len(dicts_models)):
        batch_size = dicts_models[i]['batch_size']
        split_times = num_reference // batch_size

        list_tmp = []
        for j in range(split_times):
            print('converting model {0} batch {1} ...'.format(i, j))
            sl = slice(j * batch_size, (j + 1) * batch_size)
            background_sl = background[sl]
            e = shap.DeepExplainer(dicts_models[i]['model'], background_sl)  #it will take 10 seconds
            # ...or pass tensors directly
            # e = shap.DeepExplainer((model1.layers[0].input, model1.layers[-1].output), background)
            list_tmp.append(e)

            print('converting model {0} batch {1} completed'.format(i, j))
        e_list.append(list_tmp)

    return e_list