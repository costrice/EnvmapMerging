import os
from typing import List, Tuple

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
    write_image, linrgb2srgb, srgb2linrgb,
    convert_to_uint8, convert_to_uint16
)


# transformation matrix from XYZ color space to linear sRGB color space
xyz2srgb_mat = np.array(
    [[3.24045, -1.53714, -0.49853],
     [-0.96927, 1.87601, 0.04156],
     [0.05564, -0.20403, 1.05723]], dtype=np.float32)

# transformation matrix from camera raw color space to XYZ color space
# of sony ILCE-7RM3 camera
sony_cam2xyz_mat = np.array(
    [[0.66509, 0.25285, 0.03253],
     [0.26127, 0.96595, -0.22722],
     [0.03422, -0.20912, 1.26374]], dtype=np.float32)

# of ricoh theta z1 camera
ricoh_cam2xyz_mat = np.array(
    [[0.65183, 0.05795, 0.24069],
     [0.20793, 0.84701, -0.05494],
     [-0.01642, -0.59722, 1.70248]], dtype=np.float32
)

# calibrated raw color transformation matrix from ricoh raw to sony raw
ricoh2sony_mat = np.load("./results/ricoh2sony_mat.npy").transpose()

# calibrated raw color transformation matrix from sony raw to ricoh raw
sony2ricoh_mat = np.linalg.inv(ricoh2sony_mat)

# make min value equals one
white = np.array([1, 1, 1], dtype=np.float32)
id_mat = np.diag(white)
sony2sony_mat = np.diag(white) / np.min(white@ricoh2sony_mat.transpose())
ricoh2sony_mat /= np.min(white@ricoh2sony_mat.transpose())

ricoh2ricoh_mat = np.diag(white) / np.min(white@sony2ricoh_mat.transpose())
sony2ricoh_mat /= np.min(white@sony2ricoh_mat.transpose())


def subtract_black_level(raw):
    """
    Subtract black level and demosaic.
    """
    cfa_image = raw.raw_image_visible.astype(np.int32)
    raw_colors = raw.raw_colors_visible
    # intensity range transformation
    black_level = raw.black_level_per_channel  # [4, ]
    black_level = np.array(black_level, dtype=np.float32)[raw_colors]  # [h, w]
    white_level = raw.white_level  # [, ]
    cfa_image = cfa_image - black_level
    cfa_image = cfa_image.astype(np.float32) / (white_level - black_level)
    cfa_image = np.clip(cfa_image, a_min=0, a_max=None)
    return cfa_image


def calibrate_raw_color_conversion_matrix():
    """
    Use `n` to represent the number of segments dividing visible interval of wavelength,
        `R1` in [n, 3] the Camera CFA Response function of Sony camera,
        'R2' in [n, 3] the Camera CFA Response function of Ricoh camera.
    Assume that there exists an X in [3, 3] such that R1 = R2 * X.
    Find this X using captured Standard Macbeth Chart with 24 squares.
    """
    # BEGIN: extract captured color chart using raw files
    # read captured macbeth chart raw image and manually marked chart mask image
    raw_sony = rawpy.imread("./Good/Sony/DSC07087.ARW")
    raw_ricoh = rawpy.imread("./Good/100RICOH/R0013022.DNG")
    chart_mask_sony = cv2.imread("./Good/Sony/DSC07087_MASK.PNG")
    chart_mask_ricoh = cv2.imread("./Good/100RICOH/R0013022_MASK.PNG")

    cfa_img_sony = subtract_black_level(raw_sony)
    demosaic_img_sony = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(cfa_img_sony, pattern="RGGB")
    write_image("./results/sony_demosaic_cam.png", demosaic_img_sony)

    cfa_img_ricoh = subtract_black_level(raw_ricoh)
    demosaic_img_ricoh = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(cfa_img_ricoh, pattern="RGGB")
    write_image("./results/ricoh_demosaic_cam.png", demosaic_img_ricoh)

    num_swatches = 24
    color_sony = np.zeros((num_swatches, 3), dtype=np.float32)
    color_ricoh = np.zeros((num_swatches, 3), dtype=np.float32)
    for i in range(num_swatches):
        mask_sony = (chart_mask_sony[..., 0] == (i + 1) * 5)[:, :, None]
        assert np.sum(mask_sony) > 0
        color_sony[i] = np.sum(demosaic_img_sony * mask_sony, axis=(0, 1)) / np.sum(mask_sony, axis=(0, 1))

        mask_ricoh = (chart_mask_ricoh[..., 0] == (i + 1) * 5)[:, :, None]
        assert np.sum(mask_ricoh) > 0
        color_ricoh[i] = np.sum(demosaic_img_ricoh * mask_ricoh, axis=(0, 1)) / np.sum(mask_ricoh, axis=(0, 1))

    np.save("./results/color_sony.npy", color_sony)
    np.save("./results/color_ricoh.npy", color_ricoh)
    # END: EXTRACT CAPTURED COLOR CHART USING RAW FILES

    # BEGIN: SOLVE LEAST SQUARE PROBLEM TO OBTAIN THE MATRIX
    color_sony = np.load("./results/color_sony.npy")
    color_ricoh = np.load("./results/color_ricoh.npy")
    
    ricoh2sony_matrix, residual, _, _ = np.linalg.lstsq(color_ricoh, color_sony)
    np.save("./results/ricoh2sony_mat.npy", ricoh2sony_matrix)
    # END: SOLVE LEAST SQUARE PROBLEM TO OBTAIN THE MATRIX


# def read_correction_matrices():
#     # transform raw
#     ricoh2sony_mat = np.load("./results/ricoh2sony_mat.npy")  # for ricoh
#     sony_id_mat = np.diag(np.array([1, 1, 1], dtype=np.float32))  # for sony
#     white = np.array([1, 1, 1], dtype=np.float32)
#     green = np.dot(white, ricoh2sony_mat)[1]
#     ricoh2sony_mat /= green
#     sony_id_mat /= green
#     return ricoh2sony_mat, sony_id_mat


def post_process(raw_file: str,
                 correction_mat: np.ndarray,
                 white_balance: np.ndarray,
                 cam2xyz_mat: np.ndarray,
                 return_linear: bool = False,
                 clip: Tuple[Tuple[int, int], Tuple[int, int]] = None):
    raw = rawpy.imread(raw_file)
    cfa_img = subtract_black_level(raw)
    # demosaic
    demosaic_img = colour_demosaicing.\
        demosaicing_CFA_Bayer_Menon2007(cfa_img, pattern="RGGB").astype(np.float32)
    if clip is not None:
        demosaic_img = demosaic_img[clip[0][0]: clip[0][1], clip[1][0]: clip[1][1]]
    demosaic_img = np.clip(demosaic_img, a_min=0, a_max=1)
    # raw space conversion
    demosaic_img = np.dot(demosaic_img, correction_mat.transpose())
    # white balance and exposure
    wb_img = demosaic_img * white_balance
    wb_img = np.clip(wb_img, a_min=0, a_max=1)  # over-exposured
    # cam2xyz
    xyz_img = np.dot(wb_img, cam2xyz_mat.transpose())
    # xyz to linear sRGB
    linrgb_img = np.dot(xyz_img, xyz2srgb_mat.transpose())
    linrgb_img = np.clip(linrgb_img, a_min=0, a_max=1)
    if return_linear:
        return linrgb_img
    # linear sRGB to sRGB (gamma correction)
    srgb_img = linrgb2srgb(linrgb_img)
    return srgb_img


def post_process_group(group_path):
    print(f"\nPost processing group {group_path}...")
    
    # get path and list of raw files
    envmap_path = os.path.join(group_path, "envmap_raw")
    face_path = os.path.join(group_path, "face_raw")
    raw_face_files = sorted(filter(lambda entry: entry.is_file() and entry.name.endswith(".ARW"),
                                   os.scandir(face_path)),
                            key=lambda entry: entry.name)
    raw_envmap_files = sorted(filter(lambda entry: entry.is_file() and entry.name.endswith(".DNG"),
                                     os.scandir(envmap_path)),
                              key=lambda entry: entry.name)
    
    # get white balance at shot of a raw face image in this group captured by sony camera
    ref_raw = rawpy.imread(raw_envmap_files[0].path)
    ref_white_balance = np.array(ref_raw.camera_whitebalance)[:3]  # RGB
    ref_white_balance = ref_white_balance / ref_white_balance[1]
    print(f"\nWhite balance (RGB): {ref_white_balance}")
    
    exposure = 1.5
    # # auto-exposure for face image
    # ref_img = post_process(raw_face_files[0].path,
    #                        correction_mat=sony2ricoh_mat,
    #                        white_balance=ref_white_balance,
    #                        cam2xyz_mat=ricoh_cam2xyz_mat)
    # ref_img = cv2.resize(ref_img, dsize=(797, 532))  # make computation cheaper
    # ref_img = srgb2linrgb(ref_img)
    # ref_img = ref_img[..., 0] * 0.2126 + ref_img[..., 1] * 0.7152 + ref_img[..., 2] * 0.0722
    # exposure = 0.4 / np.percentile(ref_img.reshape(-1), 90)
    # exposure = max(exposure, 1)  # must not be darker because of over-exposured region will be influenced
    print(f"Exposure for face image (in linear space): {exposure:.3f}.")
    
    # post-process sony-camera-captured face image and save as png
    for raw_face_file in tqdm(raw_face_files, desc=f"Face"):
        img = post_process(raw_face_file.path,
                           correction_mat=sony2ricoh_mat,
                           white_balance=ref_white_balance * exposure,
                           cam2xyz_mat=ricoh_cam2xyz_mat,
                           clip=((1200, 1200 + 2560), (2064, 2064 + 3840)))  # 2560 x 3840

        # denoise
        img = convert_to_uint8(img)
        img = cv2.fastNlMeansDenoisingColored(img, h=1, hColor=1, templateWindowSize=5, searchWindowSize=9)

        # save image
        save_dir = os.path.join(os.path.dirname(raw_face_file.path),
                                "../face_proc")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = f"{os.path.splitext(raw_face_file.name)[0]}.png"
        write_image(os.path.join(save_dir, save_filename), img, depth=8)
        
    # CHANGE: post-process ricoh-camera-captured envmap image by Photoshop rather than python.
    
    # post-process ricoh-camera-captured envmap image and save as tiff
    for raw_envmap_file in tqdm(raw_envmap_files, desc=f"Envmap"):
        img = post_process(raw_envmap_file.path,
                           correction_mat=id_mat,
                           white_balance=ref_white_balance,
                           cam2xyz_mat=ricoh_cam2xyz_mat,
                           return_linear=True)

        # denoise
        img = denoise_bilateral(img, sigma_spatial=1, channel_axis=-1)

        # save image
        save_dir = envmap_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = f"{os.path.splitext(raw_envmap_file.name)[0]}.tif"
        # write_image(os.path.join(save_dir, save_filename), img, depth=8)
        tifffile.imsave(os.path.join(save_dir, save_filename), convert_to_uint16(img))


def check_color_correction_matrix():
    raw_sony = rawpy.imread("./Good/Sony/DSC07087.ARW")
    raw_ricoh = rawpy.imread("./Good/100RICOH/R0013022.DNG")
    
    # fixed white balance (use sony default wb)
    white_balance = np.array(raw_ricoh.camera_whitebalance)[:3]  # RGB
    white_balance = white_balance / white_balance[1]
    
    # ricoh = post_process("./Good/100RICOH/R0013022.DNG", ricoh2ricoh_mat, white_balance, ricoh_cam2xyz_mat)
    # write_image("./results/ricoh.png", ricoh)
    sony = post_process("./Good/Sony/DSC07087.ARW", ricoh2ricoh_mat, white_balance, ricoh_cam2xyz_mat)
    write_image("./results/sony.png", sony)


if __name__ == '__main__':
    calibrate_raw_color_conversion_matrix()
    
    check_color_correction_matrix()
    
    data_root_path = "../Real-v3"
    data_groups = ["indoor-9"]  #, "outdoor-2", "outdoor-3", "indoor-1"]
    
    for data_group in data_groups:
        post_process_group(os.path.join(data_root_path, data_group))
        
    pass
