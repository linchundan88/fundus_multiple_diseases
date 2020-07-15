import pandas as pd
import sys
import os
from LIBS.DataPreprocess.my_operate_labels import get_multi_label_labelset

NUM_CLASSES = 30
IMAGE_SIZE = 299

filename_csv_train = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles',  'DLP_patient_based_split_train.csv'))
filename_csv_valid = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles',  'DLP_patient_based_split_valid.csv'))



list_samples = [0 for _ in range(NUM_CLASSES)]

df = pd.read_csv(filename_csv_train)
total_samples = len(df)

for _, row in df.iterrows():
    labels = row["labels"].strip()
    set_labels = get_multi_label_labelset(labels)

    # print(set_labels)
    for label1 in set_labels:
        list_samples[int(label1)] += 1

#label cardinality: the average number of labels per example;
total_labels = sum(list_samples)
lcard = total_labels / total_samples
print(lcard)
#label density normalizes label cardinality by the number of
# possible labels in the label space
lden = lcard / 30
print(lden)

exit(0)

#imbalance ratio (IR), defined as the ratio of the number of instances in the majority class to the number of examples in the minority class
# [49312, 14904, 3670, 204, 7339, 1283, 4704, 1877, 667, 5946, 1225, 150, 1070, 151, 135, 1461, 1658, 515, 529, 226, 2390, 8488, 1998, 323, 1285, 1182, 1408, 5739, 2045, 20119]
min_samples = min(list_samples)
max_samples = max(list_samples)

print('max/min',  max_samples / min_samples)
print('total/min',  total_samples / min_samples)
print('total/max',  total_samples / max_samples)

# min/max 0.002737670343932511
# min/total 0.0010442450495049506
# max/total 0.38143564356435644


# max/min 365.27407407407406
# total/min 957.6296296296297
# total/max 2.6216742375081115
