import numpy as np
import pickle

#region class exclusion
def _my_softmax(list_prob, list_threthold):
    max_element = max(list_prob)
    max_index = np.argmax(list_prob)

    for i, prob1 in enumerate(list_prob):
        if prob1 <= list_threthold[i]:
            continue
        if i == max_index:
            continue

        list_prob[i] = round(prob1 / (max_element+prob1), 2)

    return list_prob

def __do_exclusion(probs, list_threshold):
    match_exclusion = [[1, 2], [4, 5], [4, 7], [5, 6], [6, 8]]
    for classes_match in match_exclusion:
        num_positive = 0
        for class1 in classes_match:
            if probs[class1] > list_threshold[class1]:
                num_positive += 1

        if num_positive > 1:
            probs[classes_match[0]], probs[classes_match[1]] =\
                _my_softmax([probs[classes_match[0]], probs[classes_match[1]]],
                            list_threthold=[list_threshold[classes_match[0]], list_threshold[classes_match[1]]])

    # 10 vs 12 vs 13 vs 14
    num_positive = 0
    for class1 in [10, 12, 13, 14]:
        if probs[class1] > list_threshold[class1]:
            num_positive += 1

    if num_positive > 1:
        probs[10], probs[12], probs[13], probs[14] =\
            _my_softmax([probs[10], probs[12], probs[13], probs[14]],
                        list_threthold=[list_threshold[10], list_threshold[12], list_threshold[13], list_threshold[14]])

    return probs

def postprocess_exclusion(list_probs, list_threshold):
    if isinstance(list_probs, list):
        list_probs = np.array(list_probs)

    if list_probs.ndim == 1:  # one image's probabilities
        probs = __do_exclusion(list_probs, list_threshold)
        return probs

    if list_probs.ndim == 2:  # probabilities of a list of images
        for i in range(len(list_probs)):
            list_probs[i] = __do_exclusion(list_probs[i], list_threshold)

        return list_probs

#endregion

#region all negative
THRETHOLD_ALLNEGATIVE_CLASS0 = 0.45

def __do_all_negative(probs, list_threshold):
    num_positive = 0

    for i, prob in enumerate(probs):
        if prob > list_threshold[i]:
            num_positive += 1

    if num_positive == 0 and probs[0] < THRETHOLD_ALLNEGATIVE_CLASS0:
        class1 = np.argmax(probs)
        probs[class1] = 0.51

    return probs

def postprocess_all_negative(list_probs, list_threshold):
    if isinstance(list_probs, list):
        list_probs = np.array(list_probs)

    if list_probs.ndim == 1:  # one image's probabilities
       return __do_all_negative(list_probs, list_threshold)

    if list_probs.ndim == 2: # probabilities of a list of images
        for i in range(len(list_probs)):
            list_probs[i] = __do_all_negative(list_probs[i], list_threshold)

        return list_probs

#endregion

#region both non-referable label and referable label are positive

def _do_multi_positive(probs, list_threshold):
    num_positive = 0

    for i, prob in enumerate(probs):
        if prob > list_threshold[i]:
            num_positive += 1

    if num_positive > 0 and probs[0] > THRETHOLD_ALLNEGATIVE_CLASS0:
        if np.argmax(probs) == 0:
            for i, prob in enumerate(probs):
                if prob > list_threshold[i]:
                    probs[0] = 0.49
        else:
            probs[0] = 0.49

    return probs

def postprocess_multi_positive(list_probs, list_threshold):
    if isinstance(list_probs, list):
        list_probs = np.array(list_probs)

    if list_probs.ndim == 1:  # one image's probabilities
       return _do_multi_positive(list_probs, list_threshold)

    if list_probs.ndim == 2: # probabilities of a list of images
        for i in range(len(list_probs)):
            list_probs[i] = _do_multi_positive(list_probs[i], list_threshold)

        return list_probs

#endregion

def convert_pkl(pkl_file):
    with open(pkl_file, 'rb') as file:
        prob_total = pickle.load(file)

    prob_total_converted = postprocess_exclusion(prob_total)
    with open(pkl_file, 'wb') as file:
        pickle.dump(prob_total_converted, file)
        print('dump complete.')
