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
import queue
from tqdm import tqdm
from skimage.restoration import denoise_bilateral

from general_utils import (
    write_image,
    read_image,
    to_uint16,
    to_float32
)


def get_distance_to_border(mask: np.ndarray):
    mask = mask > 0.5
    dist = np.zeros_like(mask, dtype=np.float32)
    h, w = mask.shape
    q = []
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                q.append((i, j))
    
    d = 1
    while q:
        q2 = []
        mask2 = mask.copy()
        while q:
            i, j = q.pop(0)
            if not mask[i - 1, j] or not mask[i + 1, j] or not mask[i, j + 1] or not mask[i, j - 1]:  # at border
                dist[i, j] = d
                mask2[i, j] = False
            else:
                q2.append((i, j))
        q = q2
        mask = mask2
        d += 1
        
    return dist
            
    
def merge_group(group_path):
    print(f"\nMerging group {group_path}...")
    # directory
    envmap_dir = os.path.join(group_path, "envmap_raw")
    save_dir = os.path.join(os.path.dirname(envmap_dir), "envmap_proc")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # read list of ldr images of different exposure
    # exposures = [1 / 25000, 1 / 12500, 1 / 6400, 1 / 3200, 1 / 1600, 1 / 800, 1 / 400]  # outdoor-1, 2
    # exposures = [1 / 25000, 1 / 6400, 1 / 1600, 1 / 800, 1 / 400, 1 / 200, 1 / 100]  # outdoor 5, 6
    exposures = [1 / 25000, 1 / 12500, 1 / 6400, 1 / 4000, 1 / 2500, 1 / 640, 1 / 320, 1 / 80]  # outdoor 8, 9
    ldr_image_files = sorted(filter(lambda e: e.is_file() and e.name.endswith("_er.tif"),
                                    os.scandir(envmap_dir)),
                             key=lambda entry: entry.name)
    assert len(exposures) == len(ldr_image_files)
    
    # height, width = 3648, 7296
    height, width = 1024, 2048
    img_sum = np.zeros(shape=(height, width, 3), dtype=np.float32)
    weight_sum = np.zeros(shape=(height, width, 1), dtype=np.float32)
    for idx, ldr_image_file in enumerate(ldr_image_files):
        ldr_image = tifffile.imread(ldr_image_file.path)
        ldr_image = to_float32(ldr_image)
        ldr_image = cv2.resize(ldr_image, dsize=(width, height), interpolation=cv2.INTER_AREA)
        ldr_image[-70:, :] = 0
        if idx == 0:  # darkest
            multiplier = 1 / ldr_image.max()
        ldr_image = (ldr_image * multiplier).clip(max=1)
        
        brightness_image = ldr_image[..., 0] * 0.2126 + \
                           ldr_image[..., 1] * 0.7152 + \
                           ldr_image[..., 2] * 0.0722
        weight_image = np.exp(-4 * ((brightness_image - 0.5) ** 2) / (0.5 ** 2))
        weight_image = weight_image[:, :, None]
        
        if idx != len(ldr_image_files) - 1:
            weight_image[brightness_image < 0.05] = 0
        else:
            weight_image[brightness_image < 0.2] = 1
        if idx != 0:
            weight_image[brightness_image > 0.95] = 0
        else:
            weight_image[brightness_image > 0.7] = 1
            # find color
            aura_mask = np.bitwise_and(brightness_image < 0.95, brightness_image > 0.4).astype(np.float32)[:, :, None]
            # aura_mask[:, :1024] = 0
            write_image(os.path.join(save_dir, f"aura.png"), aura_mask * ldr_image)
            color = np.sum(aura_mask * ldr_image, axis=(0, 1)) / np.sum(aura_mask)
            color /= color.min()
            min_channel = color.argmin()
            max_channel = color.argmax()
            print(f"light color = {color}, max_c = {max_channel}, min_c = {min_channel}")
            with open(os.path.join(save_dir, f"light_color.txt"), "w") as f:
                f.write(str(color))
            sun_region = ldr_image[:, :, min_channel] > (1 / color[max_channel])
            # dist_to_border = get_distance_to_border(sun_region)
            # exposure_multiplier = 1.8 ** dist_to_border
            # multiplier = 1
            # ldr_image[sun_region] = ldr_image[sun_region][:, min_channel, None] * color * multiplier
            # ldr_image *= exposure_multiplier[:, :, None]
            
        img_sum += ldr_image / exposures[idx] * weight_image
        weight_sum += weight_image

        write_image(os.path.join(save_dir, f"{os.path.splitext(ldr_image_file.name)[0]}.png"), ldr_image)
        write_image(os.path.join(save_dir, f"{os.path.splitext(ldr_image_file.name)[0]}_w.png"), weight_image)
        
    hdr_image = img_sum / (weight_sum + 1e-6)
    hdr_image = hdr_image / hdr_image.max()
    # hdr_image = cv2.resize(hdr_image, dsize=(256, 128), interpolation=cv2.INTER_AREA)

    # adjust brightness to make average brightness = 0.3
    brightness_image = hdr_image[..., 0] * 0.2126 + \
                       hdr_image[..., 1] * 0.7152 + \
                       hdr_image[..., 2] * 0.0722
    average_brightness = np.mean(brightness_image)
    hdr_image = hdr_image / average_brightness * 0.3
    
    # save results
    write_image(os.path.join(save_dir, "envmap.hdr"), hdr_image)


def check_sun(envmap_path):
    print(f"\nChecking {envmap_path}:")
    save_dir = os.path.join(os.getcwd(), "examples")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    envmap_name = os.path.splitext(os.path.basename(envmap_path))[0]
    envmap = read_image(envmap_path)
    envmap = cv2.resize(envmap, dsize=(256, 128), interpolation=cv2.INTER_AREA)
    # darkest = to_float32(tifffile.imread(ldr_image_files[0].path))
    brt = envmap[..., 0] * 0.2126 + envmap[..., 1] * 0.7152 + envmap[..., 2] * 0.0722
    max_brightness = np.max(brt)
    sun_position = np.unravel_index(np.argmax(brt), shape=brt.shape)
    # sun_mask = (brt > (max_brightness * 0.1))[:, :, None]
    sun_mask = np.zeros_like(envmap)
    sun_mask[sun_position[0] - 2: sun_position[0] + 2, sun_position[1] - 2: sun_position[1] + 2] = 1
    sun_color = np.mean(sun_mask * envmap, axis=(0, 1))
    sun_color /= sun_color.min()
    write_image(os.path.join(save_dir, f"{envmap_name}.hdr"), envmap)
    write_image(os.path.join(save_dir, f"{envmap_name}_sun.hdr"), sun_mask * envmap)
    print(f"Max Intensity = {max_brightness}")
    print(f"Sun Color = {sun_color}")
    print(f"Sun Energy = {np.sum(sun_mask * envmap) / np.sum(envmap) * 100:.2f}%")
    
    
if __name__ == '__main__':
    # calibrate_raw_color_conversion_matrix()
    
    # check_color_correction_matrix()
    
    data_root_path = "../Real-v3"
    data_group = "outdoor-8"
    merge_group(os.path.join(data_root_path, data_group))

    check_sun(r"D:\dataset\mid-res-envmaps\outdoorPanosExr\train\9C4A0055 Panorama_hdr-P.hdr")
    check_sun(r"D:\dataset\mid-res-envmaps\outdoorPanosExr\train\9C4A9928 Panorama_hdr-P.hdr")
    check_sun(r"D:\dataset\mid-res-envmaps\outdoorPanosExr\train\9C4A0831 Panorama_hdr-P.hdr")
    check_sun(r"E:\dataset\Real-v3\outdoor-8\envmap_proc\envmap_before.hdr")
    check_sun(r"E:\dataset\Real-v3\outdoor-8\envmap_proc\envmap.hdr")
    
    # for data_group in data_groups:
    #     merge_group(os.path.join(data_root_path, data_group))
    
    pass

