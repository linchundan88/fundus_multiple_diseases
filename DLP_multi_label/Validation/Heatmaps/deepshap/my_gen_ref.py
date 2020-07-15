'''
generating reference data1
'''

import os, sys
import numpy as np
import pandas as pd
import sklearn.utils
from LIBS.ImgPreprocess import my_images_aug

filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", "..",
            'datafiles/dataset2', 'Grade_split_patid_train.csv'))

df = pd.read_csv(filename_csv)
df = sklearn.utils.shuffle(df, random_state=22222)

SAMPLES_NUM = 64
ADD_BLACK_INTERVAL = 8

imagefiles = df[0:SAMPLES_NUM - 1]['images'].tolist()

image_size = 299
x_train = my_images_aug.imgs_aug(list_image_files=imagefiles, train_or_valid='valid',
         image_shape=(image_size, image_size, 3))

#add black images
img_black = np.zeros((image_size, image_size, 3))
for i in range(SAMPLES_NUM):
    if (i%ADD_BLACK_INTERVAL == 0):
        x_train.insert(i, img_black)
# x_train = x_train[:SAMPLES_NUM]  #clip to sample_num after add black images

x_train = np.asarray(x_train, dtype=np.float16)
from LIBS.ImgPreprocess.my_image_norm import input_norm

x_train = input_norm(x_train)
# x_train /= 255.
# x_train -= 0.5
# x_train *= 2.
# (samples_num, image_size, image_size, 3)

save_filename = 'ref.npy'
np.save(save_filename, x_train)

background = np.load(save_filename)

print('OK')

