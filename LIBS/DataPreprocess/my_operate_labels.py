
#convert big class labels, convert multi labels(class 0 is special)

def get_multi_label_bigclasses(str_labels):
    label = str_labels

    label = label.replace('.0', '')
    label = label.replace('.1', '')
    label = label.replace('.2', '')
    label = label.replace('.3', '')

    label_result = convert_multilabels(label)

    return label_result


def convert_multilabels(labels):
    set_labels = set()

    list_labels = str(labels).split('_')
    for label1 in list_labels:
        if label1 != '':
            set_labels.add(label1)

    if len(set_labels) == 0:
        return '0'

    if '0' in set_labels and len(set_labels) > 1:
        set_labels.remove('0')

    label_result = ''
    for label1 in set_labels:
        label_result += (label1 + '_')

    label_result = label_result[:-1]
    return label_result


def get_multi_label_labelset(str_labels):
    labels = str_labels

    labels = labels.replace('.0', '')
    labels = labels.replace('.1', '')
    labels = labels.replace('.2', '')
    labels = labels.replace('.3', '')

    set_labels = set()

    list_labels = str(labels).split('_')
    for label1 in list_labels:
        if label1 != '':
            set_labels.add(label1)

    if len(set_labels) == 0:
        set_labels.add('0')
        return set_labels

    if '0' in set_labels and len(set_labels) > 1:
        set_labels.remove('0')

    return set_labels

