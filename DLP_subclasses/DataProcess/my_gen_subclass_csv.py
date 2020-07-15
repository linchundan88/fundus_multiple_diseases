
import sys, os
import pandas as pd
import csv
from LIBS.DataPreprocess.my_data import get_big_classes

DO_PREPROCESS = False

dir_original = '/media/ubuntu/data1/multi_labels_2919_1_15/'
dir_preprocess = '/home/ubuntu/multi_labels_2919_1_15/preprocess384/'
if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_preprocess_dir import do_process_dir
    do_process_dir(dir_original, dir_preprocess, image_size=299)


def gen_subclass_csv(filename_csv_all, filename_csv_subclass, subclass_no, one_one_class=True):
    str_subclass_no = str(subclass_no)

    df = pd.read_csv(filename_csv_all)

    if os.path.exists(filename_csv_subclass):
        os.remove(filename_csv_subclass)

    with open(filename_csv_subclass, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for i, row in df.iterrows():
            labels = row["labels"]

            if '99' in labels:  # 99,99.0 Others, 99.1 Inapplicable
                continue

            if one_one_class:
                classes = get_big_classes(labels, remove_0=False)
                if len(classes) > 1:
                    continue

            if not labels.startswith('_'):
                labels = '_' + labels
            if not labels.endswith('_'):
                labels = labels + '_'

            if str_subclass_no == '0.1':  # Tessellated fundus
                if "_0.1_" in labels:
                    csv_writer.writerow([row["images"], 1])
                elif "_0.0_" in labels:
                    csv_writer.writerow([row["images"], 0])

            elif str_subclass_no == '0.2':  # Large optic cup
                if "_0.2_" in labels:
                    csv_writer.writerow([row["images"], 1])
                elif "_0.0_" in labels:
                    csv_writer.writerow([row["images"], 0])

            elif str_subclass_no == '0.3':  # DR1
                if "_0.3_" in labels:
                    csv_writer.writerow([row["images"], 1])
                elif "_0.0_" in labels:
                    csv_writer.writerow([row["images"], 0])

            else:
                if "_" + str_subclass_no + "_" in labels:
                    # csv_writer.writerow([row["images"], 0])
                    print('No subclass label:', labels)
                    continue
                elif "_" + str_subclass_no + ".0_" in labels:
                    csv_writer.writerow([row["images"], 0])
                elif "_" + str_subclass_no + ".1_" in labels:
                    csv_writer.writerow([row["images"], 1])


# filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles',  'DLP_single_class.csv'))
filename_csv = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles',  'DLP_single_class_contain_subclass.csv'))


for subclass_no in [0.3]:
    filename_csv_subclass = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_{0}.csv'.format(subclass_no)))
    gen_subclass_csv(filename_csv, filename_csv_subclass, subclass_no=subclass_no)

# for subclass_no in [0.1, 1, 2, 5, 15, 29]:
#     filename_csv_subclass = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_{0}.csv'.format(subclass_no)))
#     gen_subclass_csv(filename_csv, filename_csv_subclass, subclass_no=subclass_no)

# for subclass_no in [10]:
#     filename_csv_subclass = os.path.abspath(os.path.join(sys.path[0], "..", 'datafiles', 'DLP_SubClass_{0}.csv'.format(subclass_no)))
#     gen_subclass_csv(filename_csv, filename_csv_subclass, subclass_no=subclass_no, one_one_class=True)


print('OK')

