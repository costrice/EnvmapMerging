import os

import cv2
import cv2.mcc
import numpy as np
import PIL
from PIL import Image
import rawpy
from matplotlib import pyplot as plt
import colour_demosaicing
import tifffile
from tqdm import tqdm
from skimage.restoration import denoise_bilateral

from ref.utils import (
    read_image,
    write_image, linrgb2srgb, srgb2linrgb,
    convert_to_uint8, convert_to_uint16,
    convert_to_float32
)


def correct_light_color(group_path):
    print(f"\nCorrecting light color for group {group_path}...")
    
    # read list of ldr images
    envmap_dir = os.path.join(group_path, "envmap_proc")
    envmap = read_image(os.path.join(envmap_dir, "envmap.hdr"))
    
    envmap_max_value = np.unravel_index(np.argmax(envmap, axis=None), envmap.shape)
    pass


if __name__ == '__main__':
    data_root_path = "../Real-v3"
    data_group = "outdoor-1"
    correct_light_color(os.path.join(data_root_path, data_group))