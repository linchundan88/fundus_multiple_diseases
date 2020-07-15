
from LIBS.ImgPreprocess.my_labelme import convert_json_mask

convert_json_mask('/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/original/images',
                  '/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/original/masks')

from LIBS.ImgPreprocess.my_rop import resize_images_dir_rop

# (640,480)->(640,512), (1600,1200)->(640,512)
resize_images_dir_rop('/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/original',
                  '/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/preprocess')

from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
resize_images_dir(source_dir='/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/preprocess',
                  dest_dir='/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/preprocess384',
                  convert_image_to_square=True, image_size=384)


import os
from LIBS.DataPreprocess.my_data import write_csv_img_seg

filename_csv = os.path.abspath('ROP1.csv')
write_csv_img_seg(filename_csv,
                '/media/ubuntu/data1/公开数据集/OpticDiscDetection/ROP1/preprocess384/images')

print('OK')
