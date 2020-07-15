import cv2
import numpy as np
from LIBS.ImgPreprocess.my_image_norm import input_norm

'''
# or tf.extract_image_patches(images,ksizes, strides,rates,padding,name=None)

image:输入图像的tesnsor，必须是[batch, in_rows, in_cols, depth]类型
ksize:滑动窗口的大小，长度必须大于四

strides:每块patch区域之间中心点之间的距离，必须是: [1, stride_rows, stride_cols, 1].

rates:在原始图像的一块patch中，隔多少像素点，取一个有效像素点，必须是[1, rate_rows, rate_cols, 1]

padding:有两个取值，“VALID”或者“SAME”，“VALID”表示所取的patch区域必须完全包含在原始图像中."SAME"表示

可以取超出原始图像的部分，这一部分进行0填充。

tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 5, 5, 1], 
    rates=[1, 1, 1, 1], padding='VALID')

'''
def extract_patch_non_overlap(full_imgs, patch_h=64, patch_w=64):
    if isinstance(full_imgs, str):
        full_imgs = cv2.imread(full_imgs)

    img_h, img_w = full_imgs.shape[:-1]

    list1 = []

    # cv2.imwrite('full.jpg', full_imgs)
    for x in range(img_w // patch_w):
        for y in range(img_h // patch_h):
            # img2 = full_imgs[x * patch_w: (x+1) * patch_w, y * patch_h: (y+1) * patch_h, :]
            img1 = full_imgs[y * patch_h: (y+1) * patch_h, x * patch_w: (x+1) * patch_w, :]

            list1.append(img1)

            # filename = "/tmp4/h{}_w{}.jpg".format(y, x)
            # print(filename)
            # cv2.imwrite(filename, img1)

    return list1

# input an image, outputa for model.predict(x_valid)
def gen_patch_data(img_file, patch_h=64, patch_w=64):

    if isinstance(img_file, str):
        image1 = cv2.imread(img_file)
    else:
        image1 = img_file

    img_height, img_width = image1.shape[:-1]
    row_num = img_height // patch_h
    col_num = img_width // patch_w

    list_patches = extract_patch_non_overlap(image1, patch_h=patch_h, patch_w=patch_w)

    list_tmp = []
    for patch1 in list_patches:
        patch1 = patch1[:, :, 1]  # G channel
        patch1 = np.expand_dims(patch1, axis=-1)
        patch1 = np.asarray(patch1, dtype=np.float16)
        patch1 = input_norm(patch1)

        list_tmp.append(patch1)

    x_valid = np.array(list_tmp)

    return x_valid

def gen_seg_result(y, img_height, img_width, threshold=127, min_size=10):
    y *= 255

    patch_h, patch_w = y[0].shape[:-1]
    row_num = img_height // patch_h
    col_num = img_width // patch_w

    img1 = np.zeros((img_height, img_width, 3))

    for index in range(y.shape[0]):
        img_patch = y[index]

        filename = "/tmp4/{}.jpg".format(index)
        cv2.imwrite(filename, img_patch)

        col = index // row_num
        row = index % row_num

        img1[row * patch_h:(row + 1) * patch_h,
        col * patch_w:(col + 1) * patch_w, :] = img_patch

    ret, binary = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)

    if min_size == 0:
        return binary
    else:
        binary = binary.astype(np.uint8)
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary[:,:,0], connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 50

        # your answer image
        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255

        return img2



if __name__ == '__main__':

    img_file = '/tmp4/img1.png'
    img1 = cv2.imread(img_file) # h:512, w:768

    img_height, img_width = img1.shape[:-1]
    img_empty = np.zeros((512, 768, 3))

    patch_h = 32
    patch_w = 64

    list1 = extract_patch_non_overlap(img_file, patch_h=patch_h, patch_w=patch_w)

    for index, img11 in enumerate(list1):
        filename = "/tmp4/{}.jpg".format(index)
        cv2.imwrite(filename, img11)

    # from top to bottom
    row_num = img_height // patch_h
    col_num = img_width // patch_w

    for index, patch1 in enumerate(list1):
        col = index // row_num
        row = index % row_num

        img_empty[row * patch_h:(row+1) * patch_h,
            col * patch_w:(col+1)*patch_w, :] =patch1


    cv2.imwrite('a.png', img_empty)

    print('OK')

