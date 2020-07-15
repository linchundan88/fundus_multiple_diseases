
'''
  Read csv file, copy subclass10 to dir
  Preprocess 512
  Crop optic disc
'''

import sys, os
from LIBS.DataPreprocess import my_data
from LIBS.DLP.my_dlp_helper import  crop_optic_disc_dir


COPY_FILES_FROM_CSV = True
DO_PREPROCESS = True
CROP_OPTIC_DISC = True
GEN_CSV = False
GET_LABELS_FROM_DIR = False

filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_10.csv'))
dir_original = '/media/ubuntu/data1/测试集_已标注/original'
dir_preprocess512 = '/media/ubuntu/data1/测试集_已标注/preprocess512'

if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess512, image_size=512)
    print('preprocess OK!')


dir_crop_optic_disc = '/media/ubuntu/data1/测试集_已标注/Crop_optic_disc_112/'

if CROP_OPTIC_DISC:
    crop_optic_disc_dir(dir_source=dir_preprocess512, dir_dest=dir_crop_optic_disc, server_port=21000, mask=True)
    print('crop optic disc 112 OK!')

if GEN_CSV:
    if GET_LABELS_FROM_DIR:
        dict_mapping = {}
        for i in range(30):
            dict_mapping[str(i)] = str(i)

        my_data.write_csv_based_on_dir(filename_csv, dir_crop_optic_disc, dict_mapping)
    else:
        my_data.write_csv_dir_nolabel(filename_csv, dir_crop_optic_disc)

print('OK')