import numpy as np
import cv2


def smallest_size_at_least(height, width, resize_min):
    smaller_dim = min(height, width)
    scale_ratio = resize_min / smaller_dim
    new_height = int(height * scale_ratio)
    new_width = int(width * scale_ratio)
    return new_height, new_width


def resize_image(image, height, width):
    return cv2.resize(image, (width, height))


def aspect_preserving_resize(image, resize_min):
    height, width = image.shape[0], image.shape[1]
    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return resize_image(image, new_height, new_width)


def central_crop(image, crop_height, crop_width):
    height, width = image.shape[0], image.shape[1]
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return image[crop_top:crop_top + crop_height,
                 crop_left:crop_left + crop_width]


def normalization(image, means):
    """
    Normalization with given means
    :param image: OpenCV Mat
    :param means: [R, G, B] ,e.g, [103.939, 116.779, 123.68]
    :return: OpenCV Mat
    """
    image = image.astype(np.float)
    image[:, :, 0] -= means[0]
    image[:, :, 1] -= means[1]
    image[:, :, 2] -= means[2]
    return image


def byte_to_mat(image_bytes, dtype):
    if isinstance(image_bytes, str):
        image_array = np.fromstring(image_bytes, dtype=dtype)
    else:
        image_array = np.frombuffer(image_bytes, dtype=dtype)
    # cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return cv2.imdecode(image_array, -1)


def NHWC2HCHW(image):
    # NHWC -> NCWH
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image
