import os,sys
from LIBS.DataPreprocess import my_data

DO_PREPROCESS = False

GEN_CSV = True

TRAIN_TYPE = 'ocular_surface'

from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
dir_original = '/media/ubuntu/data1/眼底眼表其他'
dir_preprocess = '/media/ubuntu/data1/眼底眼表其他'

# dir_original ='/media/ubuntu/data1/眼底眼表其他/tmp/original'
# dir_preprocess ='/media/ubuntu/data1/眼底眼表其他/tmp/preprocess'
# dir_original = '/media/ubuntu/data2/无法归类/original'
# dir_preprocess = '/media/ubuntu/data2/无法归类/preprocess'

if DO_PREPROCESS:
    resize_images_dir(dir_original, dir_preprocess, imgsize=299)


if GEN_CSV:
    filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '.csv'))

    dict_mapping = {'0.fundus': 0, '1.ocular_surface': 1, '2.other_images': 2}

    # 读取目录，根据目录名提取类别，生成.csv文件
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    my_data.write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping, match_type='header')

    train_files, train_labels, valid_files, valid_labels = my_data.split_dataset(
        filename_csv, valid_ratio=0.15, random_state=1111)

    filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '_train.csv'))
    filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '_valid.csv'))

    my_data.write_images_labels_csv(train_files, train_labels, filename_csv=filename_csv_train)
    my_data.write_images_labels_csv(valid_files, valid_labels, filename_csv=filename_csv_valid)


print('OK')