'''do_predict_dir()调用多个模型，预测目录中的每一个文件，返回总该率和总预测类别
虽然预测目录不需要csv文件，但是为了和文件处理一致，还是使用同一个csv文件
'''

import os
from LIBS.DataPreprocess import my_data
import pandas as pd
import shutil
import heapq

# input models and files output probs, preds


'''
  top 1 class/  top1 prob 
  for example: /0/prob35#aaaa.jpg
'''
def op_files_multiclass(all_files, prob_total, dir_preprocess,
                    dir_original='', dir_dest='/tmp', keep_subdir=False):

    if isinstance(all_files, str):  #csv file
        all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=all_files)

    for i in range(len(all_files)):
        #region  get top class, and prob of top class
        prob_list = prob_total[i].tolist()

        top_class_n = heapq.nlargest(5, range(len(prob_total[i])), prob_total[i].take)
        #top_n[0]  Softmax argmax  class no
        prob_top_class0 = round(prob_list[top_class_n[0]] * 100, 0)
        prob_top_class0 = int(prob_top_class0)
        #endregion

        img_file_source = all_files[i]

        filename = os.path.basename(img_file_source)

        if '#' in filename:    #obtain filename after #,
            filename = filename.split('#')[-1]

        filename_dest = 'prob' + str(prob_top_class0) + '#' + filename

        if keep_subdir:
            basedir = os.path.dirname(img_file_source)
            if not dir_preprocess.endswith('/'):
                dir_preprocess += '/'
            if not basedir.endswith('/'):
                basedir += '/'

            img_file_dest = os.path.join(dir_dest, basedir.replace(dir_preprocess, ''),
                                         str(top_class_n[0]), filename_dest)
        else:
            img_file_dest = os.path.join(dir_dest, str(top_class_n[0]), filename_dest)


        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        # copy original files instead of preprocessed images
        if dir_original != '':
            if dir_original.endswith('/'):
                dir_original = dir_original[:-1]
            if dir_preprocess.endswith('/'):
                dir_preprocess = dir_preprocess[:-1]

            img_file_source = img_file_source.replace(dir_preprocess, dir_original)

        if not os.path.exists(img_file_source):
            raise RuntimeError(img_file_source + ' not found!')


        shutil.copy(img_file_source, img_file_dest)

        print(img_file_dest)

'''
keep one class's probability
  for example: prob23#aaa.jpg
'''
def op_files_prob_one_class(csv_file, prob_total, class_num,
                dir_preprocess, dir_original, dest_dir, keep_subdir=False):

    df = pd.read_csv(csv_file)
    count = len(df.index)

    for i in range(count):
        img_file_source = df.iat[i, 0]

        (basedir, basename) = os.path.split(img_file_source)
        (filename, extension) = os.path.splitext(basename)

        prob1 = prob_total[i][class_num]
        prob1 = str(int(prob1 *100))

        if keep_subdir:
            if not basedir.endswith('/'):
                basedir += '/'
            if not dir_preprocess.endswith('/'):
                dir_preprocess += '/'

            img_file_dest = os.path.join(dest_dir, basedir.replace(dir_preprocess, ''),
                                'prob' + prob1 + '#' + filename + extension)
        else:
            img_file_dest = os.path.join(dest_dir, 'prob' + prob1 + '#' + filename + extension)


        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        # copy original files instead of preprocessed images
        if dir_original != '':
            if dir_original.endswith('/'):
                dir_original = dir_original[:-1]

            if dir_preprocess.endswith('/'):
                dir_preprocess = dir_preprocess[:-1]

            img_file_source = img_file_source.replace(dir_preprocess, dir_original)

        if not os.path.exists(img_file_source):
            raise RuntimeError(img_file_source + ' not found!')

        shutil.copy(img_file_source, img_file_dest)

        print(img_file_dest)



def op_files_2class(csv_file, prob_total, source_dir, dest_dir):
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]

    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    df = pd.read_csv(csv_file)
    # (row1, col1) = df.shape
    count = len(df.index)

    for i in range(count):
        # 获取TopN的序号
        top_n = heapq.nlargest(5, range(len(prob_total[i])), prob_total[i].take)
        #还是获取值
        # top_n1 = heapq.nlargest(5,  prob_total[i])
        prob_list = prob_total[i].tolist()

        # 最大概率类 top_n[0]
        #最大概率类的概率值 prob_list[top_n[0]]
        p_0 = round(prob_list[top_n[0]] * 100, 0)
        p_0 = int(p_0)

        p_1 = round(prob_list[top_n[1]] * 100, 0)
        p_1 = int(p_1)

        p_2 = round(prob_list[top_n[2]] * 100, 0)
        p_2 = int(p_2)

        img_file = df.iat[i, 0]

        img_file_source = source_dir + '/' + img_file

        #img_file本身带有子目录
        filename = os.path.basename(img_file_source)
        file_dir = os.path.dirname(img_file_source)

        if '#' in filename:
            filename = filename.split('#')[-1]  #去掉以前的概率

        filename = 'class' + str(top_n[0]) + '_' + str(p_0) + '__' + \
                   'class' + str(top_n[1]) + '_' + str(p_1) + \
                   'class' + str(top_n[2]) + '_' + str(p_2) + '#' + filename

        #将原来的文件名更改（加概率等），替换成目的文件（保留原来目录结构）
        img_file_source1 = os.path.join(file_dir, filename)
        img_file_dest = img_file_source1.replace(source_dir, dest_dir)

        # 取目录与文件名, 创建目的目录
        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        print(img_file_dest)
        shutil.copy(img_file_source, img_file_dest)


'''
保留原来的目录，文件名前面两个概率，两份类才使用
例如用DR0_DR2的模型去分 已经根据目录标注的messidor2
'''
def op_files_2class_DR(csv_file, prob_total, source_dir, dest_dir):
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]

    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    df = pd.read_csv(csv_file)
    # (row1, col1) = df.shape
    count = len(df.index)

    for i in range(count):

        p_0 = round(prob_total[i][0] * 100, 1) #0类的概率值
        p_1 = round(prob_total[i][1] * 100, 1) #1类的概率值

        img_file = df.iat[i, 0]

        img_file_source = source_dir + '/' + img_file

        #img_file本身带有子目录
        filename = os.path.basename(img_file_source)
        file_dir = os.path.dirname(img_file_source)

        if '#' in filename:
            filename = filename.split('#')[-1]  #去掉以前的概率

        filename = str(p_0) + '_' + str(p_1) + '#' + filename

        #将原来的文件名更改（加概率等），替换成目的文件（保留原来目录结构）
        img_file_source1 = os.path.join(file_dir, filename)
        img_file_dest = img_file_source1.replace(source_dir, dest_dir)

        # 取目录与文件名, 创建目的目录
        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        print(img_file_dest)
        shutil.copy(img_file_source, img_file_dest)


'''
用29个大类的模型计算DR0(0),DR2(4)
'''
def op_files_bigclasS_dr2(csv_file, prob_total, source_dir, dest_dir):
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]

    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]

    df = pd.read_csv(csv_file)
    # (row1, col1) = df.shape
    count = len(df.index)

    # 存放都同一目录，文件名前面概率值

    #region 只保留Normal和DR2的概率，并且归一
    list_2 = [[0 for col in range(2)] for row in range(len(prob_total))]
    for i in range(len(prob_total)):
        prob_bigclass0 = prob_total[i][0] / (prob_total[i][0] + prob_total[i][4])
        prob_bigclass4 = prob_total[i][4] / (prob_total[i][0] + prob_total[i][4])

        list_2[i][0] = round(prob_bigclass0, 3)
        list_2[i][1] = round(prob_bigclass4, 3)
    #endregion

    for i in range(count):
        img_file = df.iat[i, 0]
        img_file_source = os.path.join(source_dir, img_file)

        #文件名加上概率
        filename = str(list_2[i][0]) + '__' + str(list_2[i][1]) + '#' + filename

        img_file_dest = os.path.join(dest_dir, filename)

        # 取目录与文件名, 创建目的目录
        (temp_filepath, temp_filename) = os.path.split(img_file_dest)
        if not os.path.exists(temp_filepath):
            os.makedirs(temp_filepath)

        print(img_file_dest)
        shutil.copy(img_file_source, img_file_dest)
