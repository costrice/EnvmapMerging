"""
Tool for processing raw images.
"""
import colour_demosaicing
import numpy as np
import rawpy

SONY_DAYLIGHT_WB = np.array([2.73851, 1.0, 1.36871])
MY_DEFAULT_WB = np.array([2.3, 1.0, 1.7])
# Transformation matrix from XYZ color space to linear sRGB color space under
# D65? light
XYZ2SRGB_MAT = np.array(
    [[3.24045, -1.53714, -0.49853],
     [-0.96927, 1.87601, 0.04156],
     [0.05564, -0.20403, 1.05723]],
    dtype=float).T
# Transformation matrix from camera raw color space to XYZ color space
# of sony ILCE-7RM3 camera and ricoh theta z1 camera
SONY_CAM2XYZ_MAT = np.array(
    [[0.66509, 0.25285, 0.03253],
     [0.26127, 0.96595, -0.22722],
     [0.03422, -0.20912, 1.26374]],
    dtype=float).T
RICOH_CAM2XYZ_MAT = np.array(
    [[0.65183, 0.05795, 0.24069],
     [0.20793, 0.84701, -0.05494],
     [-0.01642, -0.59722, 1.70248]],
    dtype=float).T


def subtract_black_level(raw_img: rawpy._rawpy.RawPy) -> np.ndarray:
    """Subtract black level from raw image and change value range to [0, 1].

    Args:
        raw_img (rawpy._rawpy.RawPy): raw image object.

    Returns:
        np.ndarray: image with black level subtracted and converted to [0, 1].

    """
    cfa_img = raw_img.raw_image_visible.astype(np.int32)
    raw_colors = raw_img.raw_colors_visible
    # intensity range transformation
    black_level = raw_img.black_level_per_channel  # [4, ]
    black_level = np.array(black_level, dtype=np.float32)[raw_colors]  # [h, w]
    white_level = raw_img.white_level  # scalar
    cfa_img = cfa_img - black_level
    cfa_img = cfa_img.astype(np.float32) / (white_level - black_level)
    cfa_img = np.clip(cfa_img, a_min=0, a_max=None)
    return cfa_img


def camera_isp(raw_file: str,
               ccm: np.ndarray = None,
               white_balance: np.ndarray = None,
               cam2xyz_mat: np.ndarray = None) -> np.ndarray:
    """Apply camera ISP to a raw image and return linear sRGB image.

    Args:
        raw_file (str): the path to the raw image file.
        ccm (np.ndarray): color correction matrix applied in raw space.
        white_balance (np.ndarray): white balance applied in raw space.
        cam2xyz_mat (np.ndarray): camera-specific color conversion matrix.

    Returns:
        np.ndarray: linear sRGB image after ISP.
    """
    if ccm is None:
        ccm = np.eye(3)
    if white_balance is None:
        white_balance = MY_DEFAULT_WB
    if cam2xyz_mat is None:
        cam2xyz_mat = SONY_CAM2XYZ_MAT

    raw_img = rawpy.imread(raw_file)
    cfa_img = subtract_black_level(raw_img)
    # demosaicing
    demosaic_img = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(
        cfa_img, pattern="RGGB").astype(float).clip(min=0)
    # raw space conversion
    demosaic_img = demosaic_img @ ccm
    # white balance and exposure
    wb_img = demosaic_img * white_balance.reshape((1, 1, 3))
    wb_img = np.clip(wb_img, a_min=0, a_max=1)  # over-exposed
    # cam2xyz
    xyz_img = wb_img @ cam2xyz_mat
    # xyz to linear sRGB
    linrgb_img = xyz_img @ XYZ2SRGB_MAT
    linrgb_img = np.clip(linrgb_img, a_min=0, a_max=1)
    return linrgb_img
