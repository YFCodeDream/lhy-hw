import math


def calc_conv2d_image_shape(image_width, image_height, kernel_size, padding, stride, ceil_mode=False):
    kernel_width, kernel_height = kernel_size
    padding_width, padding_height = padding
    stride_width, stride_height = stride

    image_width = (image_width + 2 * padding_width - kernel_width) / stride_width + 1
    image_height = (image_height + 2 * padding_height - kernel_height) / stride_height + 1

    if ceil_mode:
        return math.ceil(image_width), math.ceil(image_height)
    return int(image_width), int(image_height)
