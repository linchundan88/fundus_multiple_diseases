import os
import cv2
import numpy as np
import uuid
import copy
from LIBS.ImgPreprocess.my_image_helper import image_border_padding

def seg_optic_disc(model, img_file_source, img_file_mask = None,
                   image_shape=(384, 384, 1), min_pixels=50, max_pixels=9000,
                   return_optic_disc_postition=False):

    if isinstance(img_file_source, str):
        img1 = cv2.imread(img_file_source)
    else:
        img1 = img_file_source

    assert img1.shape[:-1] == image_shape[:-1], 'image shape error'

    image_preprocess_RGB = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Inference
    results = model.detect([image_preprocess_RGB], verbose=1)

    if len(results) == 0:
        if return_optic_disc_postition:
            return (None, None, None, None)
        else:
            return (None, None)
    else:
        r = results[0]
        if len(r['rois']) == 0:
            if return_optic_disc_postition:
                return (None, None, None, None)
            else:
                return (None, None)

        confidence = r['scores']
        # if r['class_ids'][0] != 1:
        #     print('class id:', )
        np_zero = np.zeros(image_shape)
        np_ono = np.ones(image_shape)
        np_255 = 255 * np_ono

        temp_image = np.expand_dims(r['masks'][:, :, 0], axis=-1)
        img_masks = np.where(temp_image, np_255, np_zero)
        img_masks = img_masks.astype(np.uint8)

        if np.sum(img_masks == 255) < min_pixels or np.sum(img_masks == 255) > max_pixels:
            print('number of mask pixels:', np.sum(img_masks == 255) )
            if return_optic_disc_postition:
                return (None, None, None, None)
            else:
                return (None, None)

        y1, x1, y2, x2 = r['rois'][0]

        circle_center = ((x1+x2)//2, (y1+y2)//2)
        circle_diameter = (abs(x2-x1) + abs(y2-y1)) // 2
        # circle_radius = circle_diameter // 2

        # img2 = img1.astype(np.uint8)
        # cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if img_file_mask is None:
            str_uuid = str(uuid.uuid1())
            img_file_mask = os.path.join('/tmp', str_uuid + '.jpg')
        else:
            if not os.path.exists(os.path.dirname(img_file_mask)):
                os.makedirs(os.path.dirname(img_file_mask))

        cv2.imwrite(img_file_mask, img_masks)

        if return_optic_disc_postition:
            return (confidence, img_file_mask, circle_center, circle_diameter)
        else:
            return (confidence, img_file_mask)


def optic_disc_draw_circle(img_source, circle_center, circle_diameter, diameter_times=3, image_size=None):
    if isinstance(img_source, str):
        img_source = cv2.imread(img_source)
    else:
        img_source = img_source

    img_source = img_source.astype(np.uint8)

    img_draw_circle = copy.deepcopy(img_source)
    circle_posterior = int(circle_diameter * diameter_times)

    # cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.circle(img_draw_circle, circle_center, circle_posterior, (255, 0, 0), 1)

    return img_draw_circle



def crop_posterior(img_source, circle_center, circle_diameter, diameter_times=3,
                   crop_circle=True, image_size=None):
    if isinstance(img_source, str):
        img_source = cv2.imread(img_source)
    else:
        img_source = img_source

    img_source = img_source.astype(np.uint8)

    circle_posterior = int(circle_diameter * diameter_times)

    '''
    (height, width) = img_file_source.shape[:-1]
        
    if (circle_center[0] - circle_posterior) < 0:
        out_of_region_left = True
        # circle_center[0] += abs(circle_center[0] - circle_posterior)
    else:
        out_of_region_left = False

    if (circle_center[0] + circle_posterior) > width:
        out_of_region_right = True
        # circle_center[0] -= circle_center[0] + circle_posterior
    else:
        out_of_region_right = False

    if (circle_center[1] - circle_posterior) < 0:
        out_of_region_bottom = True
        # circle_center[1] += abs(circle_center[1] - circle_posterior)
    else:
        out_of_region_bottom = False

    if (circle_center[1] + circle_posterior) > height:
        out_of_region_top = True
        # circle_center[1] -= circle_center[1] + circle_posterior
    else:
        out_of_region_top = False
    '''

    add_border_padding = 200
    img_source = image_border_padding(img_source, padding_left=add_border_padding, padding_right=add_border_padding,
                    padding_top=add_border_padding, padding_bottom=add_border_padding)

    circle_center1 = (circle_center[0] + add_border_padding, circle_center[1] + add_border_padding)

    if crop_circle:
        image_circle = np.zeros(img_source.shape)
        cv2.circle(image_circle, circle_center1, circle_posterior, (255, 255, 255), -1)

        # cv2.imwrite('/tmp4/aa0.jpg', img_file_source)
        # cv2.imwrite('/tmp4/aa1.jpg', image_circle)

        image_circle = image_circle // 255
        img_source = np.multiply(image_circle, img_source)

        # cv2.imwrite('/tmp4/aa2.jpg', img_file_source)

    img_crop_optic_disc = img_source[circle_center1[1] - circle_posterior:circle_center1[1] + circle_posterior,
                          circle_center1[0] - circle_posterior:circle_center1[0] + circle_posterior, :]  # width, height, channel

    from LIBS.ImgPreprocess.my_image_helper import image_to_square
    img_crop_optic_disc = image_to_square(img_crop_optic_disc, image_size=image_size)

    return img_crop_optic_disc


