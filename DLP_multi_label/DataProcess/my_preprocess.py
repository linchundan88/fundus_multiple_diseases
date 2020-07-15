
from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir

'''

dir_original = '/media/ubuntu/data2/测试集_已标注/original'
dir_preprocess = '/media/ubuntu/data2/测试集_已标注/preprocess384'


'''

# dir_original = '/media/ubuntu/data1/multi_labels_2020_2_23/测试集外部集追加训练/original'
# dir_preprocess = '/media/ubuntu/data1/multi_labels_2020_2_23/测试集外部集追加训练/preprocess384'

# dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/original'
# dir_preprocess = '/media/ubuntu/data1/multi_labels_2919_1_15/preprocess384'

# dir_original = '/media/ubuntu/data1/multi_labels_2020_2_29/original'
# dir_preprocess = '/media/ubuntu/data1/multi_labels_2020_2_29/preprocess384/'

dir_original = '/media/ubuntu/data1/multi_labels_2020_2_29/original'
dir_preprocess = '/media/ubuntu/data1/multi_labels_2020_2_29/preprocess384/'

do_process_dir(dir_original, dir_preprocess, image_size=384)
