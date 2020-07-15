from keras import backend as K
import tensorflow as tf
import numpy as np
import keras.losses

#region Classification Metrics and Loss function

#tensorflow 1.14 add metrics SpecificityAtSensitivity
# from http://www.deepideas.net/unbalanced-classes-machine-learning/
def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


# loss function used in multi-label classification
# if y_true == 1 :weights[:, 1]**(y_true)
# if y_true == 0 : weights[:, 0]

def get_weighted_binary_crossentropy(weights, exclusion_loss_ratio=0):
    def weighted_loss(y_true, y_pred):
        loss_class = K.mean((weights[:, 0]**(1-y_true))*(weights[:, 1]**(y_true)) *
                      K.binary_crossentropy(y_true, y_pred), axis=-1)

        if exclusion_loss_ratio == 0:
            return loss_class
        else:
            #
            loss_exclusion = K.mean(K.sum(K.square(y_pred[:, 1] - y_pred[:, 2]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 4] - y_pred[:, 5]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 4] - y_pred[:, 7]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 5] - y_pred[:, 6]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 6] - y_pred[:, 8]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 10] - y_pred[:, 12]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 10] - y_pred[:, 13]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 10] - y_pred[:, 14]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 12] - y_pred[:, 13]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 12] - y_pred[:, 14]))) + \
                             K.mean(K.sum(K.square(y_pred[:, 13] - y_pred[:, 14])))

            # loss_exclusion = K.mean(y_pred[:, 4] / y_pred[:, 5]) + K.mean(y_pred[:, 5] / y_pred[:, 4]) + \
            #                  K.mean(y_pred[:, 4] / y_pred[:, 7]) + K.mean(y_pred[:, 7] / y_pred[:, 4]) + \
            #                  K.mean(y_pred[:, 5] / y_pred[:, 6]) + K.mean(y_pred[:, 6] / y_pred[:, 5]) + \
            #                  K.mean(y_pred[:, 6] / y_pred[:, 8]) + K.mean(y_pred[:, 8] / y_pred[:, 6]) + \
            #                  K.mean(y_pred[:, 10] / y_pred[:, 12]) + K.mean(y_pred[:, 12] / y_pred[:, 10]) + \
            #                  K.mean(y_pred[:, 10] / y_pred[:, 13]) + K.mean(y_pred[:, 13] / y_pred[:, 10]) + \
            #                  K.mean(y_pred[:, 10] / y_pred[:, 14]) + K.mean(y_pred[:, 14] / y_pred[:, 10]) + \
            #                  K.mean(y_pred[:, 12] / y_pred[:, 13]) + K.mean(y_pred[:, 13] / y_pred[:, 12]) + \
            #                  K.mean(y_pred[:, 12] / y_pred[:, 14]) + K.mean(y_pred[:, 14] / y_pred[:, 12]) + \
            #                  K.mean(y_pred[:, 13] / y_pred[:, 14]) + K.mean(y_pred[:, 14] / y_pred[:, 13])


            loss_exclusion *= -exclusion_loss_ratio  # -0.001  # -0.0220,  class:0.0650, 0.0362

            return loss_class + loss_exclusion

    return weighted_loss
#endregion


#region Segmentation Metrics and Loss function

def IOU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    iou = intersection / union

    return iou

def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0)

    if u == 0:
        return u

    return i/u

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def DICE(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

# This loss function is known as the soft Dice loss
#  because we directly use the predicted probabilities instead of
# thresholding and converting them into a binary mask.
def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    # 1-dice_coef or -dice_coef makes no difference for convergence,
    return 1.-dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def combined_loss(w_binary_crossentropy = 1, w_dice_coef_loss = 4):
    def get_loss(y_true, y_pred):

        loss_b = w_binary_crossentropy * keras.losses.binary_crossentropy(y_true, y_pred)
        loss_j = w_dice_coef_loss * dice_coef_loss(y_true, y_pred)

        return loss_b + loss_j
    return get_loss

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return 1.-jacard_coef(y_true, y_pred)

#endregion



'''
# https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras/48700950#48700950
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights
    
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def cross_entropy_balanced(y_true, y_pred):
    """
    https://github.com/lc82111/Keras_HED/blob/master/src/networks/hed.py
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits

    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    count_pos *= 7 #jijie add

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)



# https://stackoverflow.com/questions/43390162/class-weights-in-binary-classification-model-with-keras
def weighted_binary_crossentropy(weight=[1., 3.]):
    def weighted_loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        logloss = -(y_true * K.log(y_pred) * weight[1] +
                    (1 - y_true) * K.log(1 - y_pred) * weight[0])
        return K.mean(logloss, axis=-1)
    return weighted_loss

    
'''