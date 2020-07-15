import cv2
import numpy as np


img1 = cv2.imread('/media/ubuntu/data1/公开数据集/视盘/Refuge/preprocess384/Validation400/masks/V0209.bmp',
                  cv2.IMREAD_GRAYSCALE)

# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image2 = np.zeros((384, 384, 3))
# thickness != -1 ,draw line
img = cv2.drawContours(image2, [contours[0]], -1, (255, 255, 255), thickness=-1)

cv2.imwrite('/tmp2/4.bmp', image2)

print('a')