import math
from typing import Optional

import progressbar


def calc_conv2d_image_shape(image_width, image_height, kernel_size, padding, stride, ceil_mode=False):
    kernel_width, kernel_height = kernel_size
    padding_width, padding_height = padding
    stride_width, stride_height = stride

    image_width = (image_width + 2 * padding_width - kernel_width) / stride_width + 1
    image_height = (image_height + 2 * padding_height - kernel_height) / stride_height + 1

    if ceil_mode:
        return math.ceil(image_width), math.ceil(image_height)
    return int(image_width), int(image_height)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def show_download_progress_bar(block_num, block_size, total_size):
    progress_bar = progressbar.ProgressBar(maxval=total_size)
    progress_bar.start()

    downloaded_size = block_num * block_size

    if downloaded_size < total_size:
        progress_bar.update(downloaded_size)
    else:
        progress_bar.finish()
        progress_bar = None
