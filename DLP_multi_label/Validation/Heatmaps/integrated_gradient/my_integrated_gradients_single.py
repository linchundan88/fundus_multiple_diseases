
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import keras
from LIBS.Heatmaps.IntegratedGradient.IntegratedGradients import integrated_gradients
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess import my_preprocess

#Wrap it with integrated_gradients.
model_file = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_1_5/InceptionV3-015-0.993.hdf5'
model1 = keras.models.load_model(model_file, compile=False)
#because .model.optimizer.get_gradients
model1.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
               , metrics=['acc'])

image_file = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200102_修正20191119分期病变 +正常/preprocess384/广州妇幼2017-2018/分期病变/795dd342-5751-4f58-bfdb-3a7f19ffe253.11.jpg'
image_size = 299
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

ig = integrated_gradients(model1)

ig_result = ig.explain(img_input[0], outc=class_predict)
exs = []
# import time
# print(time.time())
exs.append(ig.explain(img_input[0], outc=class_predict))
# print(time.time())
# print('OK')

# Plot them
th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

plt.axis('off')
plt.imshow(exs[0][:, :, 0], cmap="seismic", vmin=-1 * th, vmax=th)
plt.savefig('/tmp3/a.jpg', bbox_inches='tight')
plt.show()
print('OK')


