import math
import numpy as np
import os
from typing import List, Tuple
import rawpy

import cv2
import colour_demosaicing

from pdb import set_trace as st

xyz2srgb_mat = np.array(
    [[3.24045, -1.53714, -0.49853],
     [-0.96927, 1.87601, 0.04156],
     [0.05564, -0.20403, 1.05723]], dtype=np.float32)


sony_cam2xyz_mat = np.array(
    [[0.66509, 0.25285, 0.03253],
     [0.26127, 0.96595, -0.22722],
     [0.03422, -0.20912, 1.26374]], dtype=np.float32)

ricoh_cam2xyz_mat = np.array(
    [[0.65183, 0.05795, 0.24069],
     [0.20793, 0.84701, -0.05494],
     [-0.01642, -0.59722, 1.70248]], dtype=np.float32
)

def convert_to_uint8(img: np.ndarray):
    """
    Convert an image into np.uint8. If float, clip to [0, 1] first.
    """
    if img.dtype == np.bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 255).astype(np.uint8)
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, a_min=0, a_max=1) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def convert_to_uint16(img: np.ndarray):
    """
    Convert an image into np.uint16. If float, clip to [0, 1] first.
    """
    if img.dtype == np.bool:
        img = img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 255
    if img.dtype == np.uint16:
        return img
    if img.dtype in [np.float32, np.float64]:
        return np.around(np.clip(img, a_min=0, a_max=1) * 65535).astype(np.uint16)
    raise ValueError(f"Unsupported dtype: {img.dtype}")
    

def convert_to_float32(img: np.ndarray):
    """
    Convert an image into np.float32
    """
    if img.dtype == np.bool:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535
    if img.dtype == np.float32:
        return img
    raise ValueError(f"Unsupported dtype: {img.dtype}")


def write_image(output_path: str, img: np.ndarray, depth: int = 8):
    """
    Write an ndarray <img> into <output_path>, LDR or HDR according to extension specified in <output_path>.
    
    Args:
        output_path: the save path.
        img: the image to be saved.
        depth: bit depth, can be 8 or 16.
    """
    if img.dtype in [np.float64, np.bool]:  # cv2 do not support float64?
        img = img.astype(np.float32)
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = cv2.cvtColor(img, code=cv2.COLOR_RGB2BGR)
    if output_path.endswith((".hdr", ".exr")):
        cv2.imwrite(output_path, convert_to_float32(img))
    elif output_path.endswith((".png", ".jpg")):
        if depth == 8:
            cv2.imwrite(output_path, convert_to_uint8(img))
        elif depth == 16:
            cv2.imwrite(output_path, convert_to_uint16(img))
        else:
            raise ValueError(f"Unexpected depth {depth}")
    else:
        raise ValueError(f"Unexpected file extension in {output_path}")


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

def post_process(raw_file: str,
                 correction_mat: np.ndarray,
                 white_balance: np.ndarray,
                 cam2xyz_mat: np.ndarray,
                 return_linear: bool = True,
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

def calibrate_intensity():
    sony_f = 3.5
    sony_iso = 16000
    sony_shutter = 1 / 1600
    # sony_image = 

    ricoh_f = 3.5
    ricoh_iso = 1600
    ricoh_shutter = 1 / 640
    ricoh_intensity = [112, 48, 23, 29, 52, 26, 91, 94, 99]

    st()

class HDRCalculator:
    def __init__(self):
        self.sony_mm_focal = 24.
        self.sony_mm_sensor_width = 36.
        self.sony_mm_sensor_height = 24.
        self.sony_pixel_width = 7952.
        self.sony_pixel_height = 5304.
        self.sony_mm_pixel_area = self.sony_mm_sensor_width * self.sony_mm_sensor_height / self.sony_pixel_width / self.sony_pixel_height

        self.ricoh_pixel_width = 6720.
        self.ricoh_pixel_height = 3360.

    def get_solid_angle(self, pixel_w, pixel_h):
        abs_w = abs(self.sony_pixel_width / 2 - pixel_w)
        abs_h = abs(self.sony_pixel_height / 2 - pixel_h)

        d = math.sqrt(abs_w ** 2 + abs_h ** 2)
        d_mm = d * self.sony_mm_sensor_width / self.sony_pixel_width
        theta = math.atan(d_mm / self.sony_mm_focal)

        # calculate solid angle
        r = self.sony_mm_focal / math.cos(theta)
        solid_angle = self.sony_mm_pixel_area / (r ** 2) * math.cos(theta)
        
        return solid_angle

    def get_ricoh_pixel_angle(self):
        # https://zhuanlan.zhihu.com/p/450731138
        dphi = 2 * math.pi / self.ricoh_pixel_width
        dtheta = math.pi / self.ricoh_pixel_height

        pixel_h = 700 # for R0013203
        theta = abs(self.ricoh_pixel_height / 2 - pixel_h) / self.ricoh_pixel_height * math.pi

        solid_angle = dphi * dtheta * math.sin(theta)

        return solid_angle

        
if __name__ == '__main__':
    # sony_image_path = '/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/CI_00727.ARW'
    # raw_sony = rawpy.imread(sony_image_path)
    # white_balance = np.array(raw_sony.camera_whitebalance)[:3]  # RGB
    # white_balance = white_balance / white_balance[1]
    # eye_mat = np.eye(3)
    # sony = post_process(sony_image_path, eye_mat, white_balance, sony_cam2xyz_mat)
    # cv2.imwrite('sony_linear.png', sony * 255.)

    sony = cv2.imread('sony_linear.png')

    sun_mask = sony[:, :, 0] > 0.005
    sun_mask = sun_mask.astype(np.float32)

    cv2.imwrite('sony_linear_mask.png', sun_mask * 255.)

    # accumulate the intensity: \Sigma sony[i, j] * solid_angle

    calc = HDRCalculator()
    out_intensity = 0
    for i in range(sony.shape[0]):
        for j in range(sony.shape[1]):
            if sun_mask[i, j] == 0:
                continue
            solid_angle = calc.get_solid_angle(i, j)
            out_intensity += sony[i, j] * solid_angle
    print(out_intensity)

    # print("Processing Sony images...")
    # sony_cali_path = '/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/CI_00738.ARW'
    # raw_sony = rawpy.imread(sony_cali_path)
    # white_balance = np.array(raw_sony.camera_whitebalance)[:3]  # RGB
    # white_balance = white_balance / white_balance[1]
    # eye_mat = np.eye(3)
    # sony_cali = post_process(sony_cali_path, eye_mat, white_balance, sony_cam2xyz_mat)
    # write_image('sony_cali.png', sony_cali)

    sony_cali = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/sony_cali.png')
    sony_mask = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/CI_00738_mask.png')
    sony_mask = cv2.resize(sony_mask, (sony_cali.shape[1], sony_cali.shape[0]))

    # st()

    # print("Processing Ricoh images...")
    # ricoh_cali_path = '/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/R0013210.DNG'
    # raw_ricoh = rawpy.imread(ricoh_cali_path)
    # # white_balance = np.array(raw_ricoh.camera_whitebalance)[:3]  # RGB
    # # white_balance = white_balance / white_balance[1]
    # eye_mat = np.eye(3)
    # ricoh_cali = post_process(ricoh_cali_path, eye_mat, white_balance, ricoh_cam2xyz_mat)
    # write_image('ricoh_cali.png', ricoh_cali)

    ricoh_cali = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/ricoh_cali.png')
    ricoh_mask = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/R0013210_mask.png')

    # calculate mean
    sony_int_mean = np.sum(sony_cali[sony_mask > 0]) / np.sum(sony_mask > 0)
    ricoh_int_mean = np.sum(ricoh_cali[ricoh_mask > 0]) / np.sum(ricoh_mask > 0)

    print(sony_int_mean, ricoh_int_mean)
    base_sony2ricoh = ricoh_int_mean / sony_int_mean


    print("Processing Sony images...")
    sony_cali_path = '/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/CI_00737.ARW'
    raw_sony = rawpy.imread(sony_cali_path)
    white_balance = np.array(raw_sony.camera_whitebalance)[:3]  # RGB
    white_balance = white_balance / white_balance[1]
    eye_mat = np.eye(3)
    sony_f22 = post_process(sony_cali_path, eye_mat, white_balance, sony_cam2xyz_mat)
    write_image('sony_f22.png', sony_f22)

    sony_f22 = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/sony_f22.png')
    sony_mask = cv2.imread('/userhome/chengyean/face_lighting/lief_face_lighting/sun_hdr/CI_00738_mask.png')
    sony_mask = cv2.resize(sony_mask, (sony_cali.shape[1], sony_cali.shape[0]))

    sony_int_mean_f22 = np.sum(sony_f22[sony_mask > 0]) / np.sum(sony_mask > 0)

    print(sony_int_mean_f22, sony_int_mean)
    base_sonyf222sony = sony_int_mean / sony_int_mean_f22

    ricoh_solid_angle = calc.get_ricoh_pixel_angle()

    # sony_shutter = 1 / 8000
    # sony_iso = 50

    # ricoh_shutter = 1 / 25000
    # ricoh_iso = 100

    int_max = out_intensity.mean() / ricoh_solid_angle * base_sony2ricoh * base_sonyf222sony * (100 / 50) * ((1 / 25000) / (1 / 8000)) * 1000

    # int_max / 3360 / 6720 * 128 * 64 = 46530.922126938924

    # st()

