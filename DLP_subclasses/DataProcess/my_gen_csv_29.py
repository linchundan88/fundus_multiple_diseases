import os,sys
from LIBS.DataPreprocess import my_data

DO_PREPROCESS = False
GEN_CSV = True

TRAIN_TYPE = 'Subclass_29'

from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
dir_original = '/media/ubuntu/data1/子类/29/original'
dir_preprocess = '/media/ubuntu/data1/子类/29/preprocess384'

if DO_PREPROCESS:
    do_process_dir(dir_original, dir_preprocess, image_size=384)

if GEN_CSV:
    filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '.csv'))

    # dict_mapping = {'1.0': 0, '1.1': 1}
    dict_mapping = {'29.0': 0, '29.1': 1}

    # 读取目录，根据目录名提取类别，生成.csv文件
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    my_data.write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping)

    train_files, train_labels, valid_files, valid_labels = my_data.split_dataset(
        filename_csv, valid_ratio=0.15, random_state=1111)

    filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '_train.csv'))
    my_data.write_images_labels_csv(train_files, train_labels, filename_csv=filename_csv_train)
    filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', TRAIN_TYPE + '_valid.csv'))
    my_data.write_images_labels_csv(valid_files, valid_labels, filename_csv=filename_csv_valid)


print('OK')