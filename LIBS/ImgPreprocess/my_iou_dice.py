import numpy as np
import cv2

def iou_dicee_bbox(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    dice = (2 * interArea) / float(boxAArea + boxBArea)

    return iou, dice


def iou_dice_mask(img1, img2, threshold = 3):

    if isinstance(img1, str):
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

    if isinstance(img2, str):
        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    # threshold = 3
    upper = 1
    lower = 0

    if img1.ndim == 2:
        img1 = img1[:, :, np.newaxis]
    if img2.ndim == 2:
        img2 = img2[:, :, np.newaxis]

    img1 = np.where(img1 > threshold, upper, lower)
    img2 = np.where(img2 > threshold, upper, lower)

    multip = np.multiply(img1, img2)
    intersection = np.sum(multip == upper * upper)

    union = np.sum(img1 == upper) + np.sum(img2 == upper) - intersection
    iou = intersection / union

    dice = (2 * intersection) / ( np.sum(img1 == upper) + np.sum(img2 == upper))

    #  dice/2 < iou < dice
    return iou, dice

