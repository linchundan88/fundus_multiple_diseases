
import sys
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from LIBS.Heatmaps.IntegratedGradient.IntegratedGradients import integrated_gradients
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess


model_file = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_1_5/InceptionV3-015-0.993.hdf5'
model1 = keras.models.load_model(model_file, compile=False)
#because .model.optimizer.get_gradients, our model use some customobject
model1.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
               , metrics=['acc'])
image_size = 299

#Wrap it with integrated_gradients.
ig = integrated_gradients(model1)

DIR_DEST = '/tmp5/ROP_Integrated_gradient/2020_1_23'
dir_preprocess = '/media/ubuntu/data1/ROP项目/preprocess384/'

for predict_type_name in ['Grade_split_patid_train', 'Grade_split_patid_valid', 'Grade_split_patid_test']:
    save_dir = os.path.join(DIR_DEST, predict_type_name)
    filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",  "..", "..",
                    'datafiles/dataset3', predict_type_name + '.csv'))
    df = pd.read_csv(filename_csv)

    for _, row in df.iterrows():
        image_file = row['images']
        image_label = int(row['labels'])

        preprocess = False
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                             image_shape=(image_size, image_size, 3))
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                             image_shape=(image_size, image_size, 3))

        prob = model1.predict(img_input)
        class_predict = np.argmax(prob)

        if (class_predict == 1 and image_label == 1) or \
                (class_predict == 1 and image_label == 0):

            if class_predict == 1 and image_label == 1:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '1_1/'))

            if class_predict == 1 and image_label == 0:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '0_1/'))

            if not os.path.exists(os.path.dirname(file_dest)):
                os.makedirs(os.path.dirname(file_dest))

            ig_result = ig.explain(img_input[0], outc=class_predict)
            exs = []
            exs.append(ig.explain(img_input[0], outc=class_predict))

            #plot image
            th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

            plt.axis('off')
            plt.imshow(exs[0][:, :, 0], cmap="seismic", vmin=-1 * th, vmax=th)
            plt.savefig(file_dest, bbox_inches='tight')
            plt.close()

print('OK')


