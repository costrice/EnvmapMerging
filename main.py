"""
This code is used to merge a set of LDR panoramas into a single HDR panorama,
then correct the clipped intensity of the sun in the panorama using an image
captured with a ND filter.
"""
import glob
import json
import os
from typing import Dict, Tuple

import colour_demosaicing
import cv2
import exifread
import numpy as np
import rawpy
import tifffile
from scipy import io
from skimage import restoration
from tqdm.auto import tqdm

from utils import extract_filename, read_image, to_float32, to_uint16, \
    write_image

DATA_DIR = os.path.abspath("./Data")
CALIB_DATA_DIR = os.path.join(DATA_DIR, "for_calibration")
BASE_EXPOSURE = {
    "ISO": 100.0,
    "shutter speed": 0.01,
    "aperture": 3.5,
}
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
SONY_CAM_PIXEL_W = 7952
SONY_CAM_PIXEL_H = 5304
SONY_CAM_SENSOR_W = 35.9  # unit: mm
SONY_CAM_SENSOR_H = 24  # unit: mm
EXPOSURE_INFO_FILENAME = "exposure_infos.json"


def read_exposure_from_raw(raw_file: str) -> Dict[str, float]:
    """Read exposure information from a raw file.

    Read exposure information (ISO, shutter speed, aperture) from EXIF
    data of a raw file, converting numbers to float.

    Args:
        raw_file (str): path to the raw image file.

    Returns:
        Dict: containing exposure information. For example:
         {"ISO": 500.0,
          "shutter speed": 0.01,
          "aperture": 2.8,}
    """
    with open(raw_file, "rb") as f:  # pylint: disable=redefined-outer-name
        exif = exifread.process_file(f, details=False)
        exposure = {
            "ISO": float(exif["EXIF ISOSpeedRatings"].values[0]),
            "shutter speed": float(exif["EXIF ExposureTime"].values[0]),
            "aperture": float(exif["EXIF FNumber"].values[0]),
        }
    return exposure


def relative_exposure_amount(exposure: Dict[str, float]) -> float:
    """Compute the relative exposure of an exposure setting.

    Compute how many times the exposure effect of the input exposure
    image is that of the standard exposure setting of the same camera.

    Args:
        exposure (Dict[str, float]): exposure information of the image.

    Returns:
        float: relative exposure.
    """
    base_exposure_amount = BASE_EXPOSURE["ISO"] * \
                           BASE_EXPOSURE["shutter speed"] / \
                           BASE_EXPOSURE["aperture"] ** 2
    input_exposure_amount = exposure["ISO"] * \
                            exposure["shutter speed"] / \
                            exposure["aperture"] ** 2
    return input_exposure_amount / base_exposure_amount


def calibrate_ccm_ricoh2sony(sony_file: str, ricoh_file: str) \
        -> Tuple[np.ndarray, float]:
    """Calibrate the color correction matrix from Ricoh to Sony.

    Calibrate the color correction matrix from Ricoh to Sony by reading the
    relative exposure and colors of the color checker patches in the two raw
    images, then solving the linear system of equations
        sony_colors / sony_relative_exposure
        = ricoh_colors / ricoh_relative_exposure * ccm / over_exposure
    to obtain the color correction matrix. The CCM is decomposed into ccm
    / over_exposure, since as a CCM it must map a white color to an over-exposed
    white color, thus the sum of its each column cannot less than 1. The
    ``over_exposure`` factor indicates how much time the ``CCM`` is enlarged to
    satisfy this constraint.
    Then, the color correction matrix can be used to convert the colors of a
    raw image captured by Ricoh camera to those of a raw image captured by Sony
    camera, both using the standard exposure setting defined by the global
    variable BASE_EXPOSURE, by simple right multiplying CCM. However, remember
    the  equivalent relative exposure of the image is increased by
    ``over_exposure``.

    Args:
        sony_file (str): path to the raw image file captured by Sony camera.
        ricoh_file (str): path to the raw image file captured by Ricoh
            camera.

    Returns:
        Tuple[np.ndarray, float]: the color correction matrix and the
        over_exposure factor.
    """
    ricoh2sony_ccm_file = os.path.join(CALIB_DATA_DIR, "ricoh2sony_ccm.mat")
    if os.path.exists(ricoh2sony_ccm_file):
        print(f"Loading Ricoh2Sony CCM from {ricoh2sony_ccm_file}...")
        return (io.loadmat(ricoh2sony_ccm_file)["ccm"],
                io.loadmat(ricoh2sony_ccm_file)["over_exposure"].item())
    else:
        # Read .ARW file captured by Sony camera
        sony_rel_exposure, sony_colors = \
            read_color_checker_from_raw(sony_file)
        # Read .DNG file captured by Ricoh camera
        ricoh_rel_exposure, ricoh_colors = \
            read_color_checker_from_raw(ricoh_file)
        # Compute CCM
        ricoh2sony_ccm, _, _, _ = \
            np.linalg.lstsq(ricoh_colors, sony_colors)
        # Consider in the effect of exposure
        ricoh2sony_ccm = ricoh2sony_ccm * ricoh_rel_exposure / sony_rel_exposure
        # Enlarge the CCM to ensure that when an input channel is saturated, the
        # output must also be saturated (>=1).
        diag_mat = np.diag(np.diag(ricoh2sony_ccm))
        worst_case = np.minimum(ricoh2sony_ccm - diag_mat, 0) + diag_mat
        over_exposure = 1 / np.min(np.sum(worst_case, axis=0))
        ricoh2sony_ccm *= over_exposure
        io.savemat(ricoh2sony_ccm_file, {"ccm": ricoh2sony_ccm,
                                         "over_exposure": over_exposure})
        return ricoh2sony_ccm, over_exposure


def calibrate_ndfilter(
        raw_file_w_filter: str,
        raw_file_wo_filter: str,
) -> Tuple[float, np.ndarray]:
    """Calibrate the ND filter by comparing two raw images captured w/wo filter.

    Calibrate the ND filter by reading the relative exposure and colors of the
    input two raw images, then solving the linear system of equations
        wo_filter_colors / wo_filter_relative_exposure
        = w_filter_colors / w_filter_relative_exposure * CCM * density
    to obtain the ND filter density and CCM (which doesn't change the total
    intensity).

    Args:
        raw_file_w_filter (str): path to the raw image file captured with ND
            filter.
        raw_file_wo_filter (str): path to the raw image file captured without ND
            filter.

    Returns:
        Tuple[float, np.ndarray]: ND filter density and CCM.
    """
    nd_filter_file = os.path.join(CALIB_DATA_DIR, "nd_filter.mat")
    if os.path.exists(nd_filter_file):
        print(f"Loading ND filter density and CCM from {nd_filter_file}...")
        return (io.loadmat(nd_filter_file)["density"].item(),
                io.loadmat(nd_filter_file)["ccm"])
    else:
        # Read .ARW file captured with ND filter
        w_filter_rel_exposure, w_filter_colors = \
            read_color_checker_from_raw(raw_file_w_filter)
        # Read .ARW file captured without ND filter
        wo_filter_rel_exposure, wo_filter_colors = \
            read_color_checker_from_raw(raw_file_wo_filter)
        # Compute CCM
        ccm, _, _, _ = \
            np.linalg.lstsq(w_filter_colors, wo_filter_colors)
        # Consider in the effect of exposure
        white = np.array([1, 1, 1], dtype=float)
        white_transformed = white @ ccm
        density = np.sum(white_transformed) / 3
        ccm = ccm / density
        density *= w_filter_rel_exposure / wo_filter_rel_exposure
        io.savemat(nd_filter_file, {"density": density, "ccm": ccm})
        return density, ccm


def detect_color_checker_masks(raw_img: rawpy._rawpy.RawPy) -> np.ndarray:
    """Detect the color checker in the raw image and generate masks of patches.

    Detect the color checker in the raw image and generate masks of patches.
    Since there are 24 patches in the color checker, the returned mask is a
    unified uint8 mask of 24 binary masks, denoting the i-th mask by a pixel
    value of (i + 1) * 10.

    Args:
        raw_img (rawpy._rawpy.RawPy): the raw image containing a color checker.

    Returns:
        np.ndarray: the mask of the individual patches in the color checker
        of size (raw_img.height, raw_img.width, 1).
    """
    # detect the color checker and find its bounding box
    proc_img = raw_img.postprocess()
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(proc_img, cv2.mcc.MCC24, nc=1)
    checker = detector.getBestColorChecker()
    box = checker.getBox()
    box_left = min(box[:, 0])
    box_right = max(box[:, 0])
    box_top = min(box[:, 1])
    box_bottom = max(box[:, 1])
    # generate masks of individual color patches
    patch_w = (box_right - box_left) / 6
    patch_h = (box_bottom - box_top) / 4
    mask = np.zeros((*proc_img.shape[:2], 1), dtype=np.uint8)
    for i in range(4):
        for j in range(6):
            color_id = i * 6 + j
            patch_center = (box_top + patch_h * (i + 0.5),
                            box_left + patch_w * (j + 0.5),)
            mask[int(patch_center[0] - patch_h / 8):
                 int(patch_center[0] + patch_h / 8),
            int(patch_center[1] - patch_w / 8):
            int(patch_center[1] + patch_w / 8)] = (color_id + 1) * 10
    return mask


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


def read_color_checker_from_raw(raw_img_file: str) -> Tuple[float, np.ndarray]:
    """ For a raw image, read the colors in the checker and relative exposure.

    Read a raw image, extract its exposure information from EXIF and compare
    to base exposure, then read the 24 color values in the captured color
    checker using the color checker mask (existing or auto-generated).
    This function is for calibration purpose.

    Args:
        raw_img_file (str): path to the raw image file.

    Returns:
        Tuple[float, np.ndarray]: relative exposure and 24 color values.
    """
    raw_img = rawpy.imread(raw_img_file)
    cfa_img = subtract_black_level(raw_img)
    demosaic_img = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(
        cfa_img, pattern="RGGB")

    ## read exposure and compute relative exposure
    exposure = read_exposure_from_raw(raw_img_file)
    rel_exposure = relative_exposure_amount(exposure)

    ## read colors in the color checker
    # read color checker mask
    raw_mask_file = os.path.join(CALIB_DATA_DIR,
                                 f"{extract_filename(raw_img_file)}_mask.png")
    if os.path.exists(raw_mask_file):
        mask = cv2.imread(raw_mask_file)
        if len(mask.shape) == 3:
            mask = mask[:, :, :1]
    else:
        mask = detect_color_checker_masks(raw_img)
    # cv2.imwrite(raw_mask_file, mask)
    # compute mean color in each patch
    colors = []
    for color_id in range(24):
        patch_mask = (mask[..., 0] == (color_id + 1) * 10)
        assert np.sum(patch_mask) > 0
        colors.append(demosaic_img[patch_mask].mean(axis=0))
    colors = np.array(colors, dtype=float)

    return rel_exposure, colors


def preprocess_envmap_group(envmap_dir, ccm, over_exposure):
    """Preprocess a group of envmap image (.DNG files).

    For a .DNG envmap file, first use the CCM to convert its color space, then
    save as a TIFF file. Save the exposure information in a JSON file for
    later use.

    Args:
        envmap_dir (str): path to the directory containing original .DNG files.
        ccm (np.ndarray): the color correction matrix from Ricoh to Sony.
        over_exposure (float): the enlarging multiplier of ccm for saving in
            the JSON file.
    """
    envmap_raw_files = glob.glob(os.path.join(envmap_dir, "*.DNG"))
    exposure_info_file = os.path.join(envmap_dir, EXPOSURE_INFO_FILENAME)
    # Check if this group has been processed
    if os.path.exists(exposure_info_file):
        all_processed = True
        for envmap_raw_file in envmap_raw_files:
            filename = extract_filename(envmap_raw_file)
            envmap_tiff_file = os.path.join(envmap_dir, f"{filename}.TIF")
            if not os.path.exists(envmap_tiff_file):
                all_processed = False
                break
        if all_processed:
            print(
                f"Envmap preprocessing: group {envmap_dir} has been processed.")
            return
    # Process the group otherwise
    exposure_infos = {}
    pbar = tqdm(envmap_raw_files, desc="Envmap")
    for envmap_raw_file in pbar:
        pbar.set_description(f"Envmap: {extract_filename(envmap_raw_file)}")
        # meta information
        filename = extract_filename(envmap_raw_file)
        # read exposure information
        exposure_info = read_exposure_from_raw(envmap_raw_file)
        exposure_infos[filename] = exposure_info
        # check if the envmap has been processed
        envmap_tiff_file = os.path.join(envmap_dir, f"{filename}.TIF")
        if os.path.exists(envmap_tiff_file):
            continue
        # process the envmap image
        envmap_linsrgb = camera_isp(envmap_raw_file,
                                    ccm,
                                    MY_DEFAULT_WB,
                                    SONY_CAM2XYZ_MAT)
        # denoise
        envmap_linsrgb = restoration.denoise_bilateral(
            envmap_linsrgb, sigma_spatial=1, channel_axis=-1)
        # save file as .TIF file
        tifffile.imsave(envmap_tiff_file, to_uint16(envmap_linsrgb))
    exposure_infos["over_exposure_coeff"] = over_exposure
    # save exposure information
    with open(exposure_info_file, "w", encoding="utf-8") as f: # pylint: disable=redefined-outer-name
        json.dump(exposure_infos, f, ensure_ascii=False, indent=4)


def camera_isp(raw_file: str,
               ccm: np.ndarray,
               white_balance: np.ndarray,
               cam2xyz_mat: np.ndarray):
    """Apply camera ISP to a raw image and return linear sRGB image.

    Args:
        raw_file (str): the path to the raw image file.
        ccm (np.ndarray): color correction matrix applied in raw space.
        white_balance (np.ndarray): white balance applied in raw space.
        cam2xyz_mat (np.ndarray): camera-specific color conversion matrix.

    Returns:
        np.ndarray: linear sRGB image after ISP.
    """
    raw_img = rawpy.imread(raw_file)
    cfa_img = subtract_black_level(raw_img)
    # demosaicing
    demosaic_img = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(
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


def ensure_envmaps_are_stitched(envmap_dir: str) -> bool:
    """Ensure that all fish-eye images have been stitched to panorama.

    Ensure that all fish-eye images, which are .DNG files, have been manually
    stitched to panoramas, which are .TIF files with suffix '_er'.

    Args:
        envmap_dir (str): path to the directory containing original .DNG files.

    Returns:
        bool: whether all fish-eye images have been stitched to panorama.
    """
    envmap_files = glob.glob(os.path.join(envmap_dir, "*.DNG"))
    for envmap_file in envmap_files:
        filename = extract_filename(envmap_file)
        if not os.path.exists(os.path.join(envmap_dir, f"{filename}_er.TIF")):
            return False
    return True


def merge_ldrs_into_hdr(envmap_dir: str) -> np.ndarray:
    """Merge LDR panorama images into HDR panorama images.

    Merge LDR panorama images, which are .TIF files with suffix '_er', into
    an HDR panorama image and save as a .HDR file. The merging is done by
    a weighted average of the LDR images, where the weights are computed using
    the reliability of pixel intensity values in each LDR image.

    Args:
        envmap_dir (str): path to the directory containing LDR panorama images.

    Returns:
        np.ndarray: the HDR panorama image.
    """
    save_dir = os.path.join(os.path.dirname(envmap_dir), "envmap")
    hdr_envmap_path = os.path.join(save_dir, "envmap.hdr")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif os.path.exists(hdr_envmap_path):
        print(f"HDR merging: group {envmap_dir} has been processed.")
        return read_image(hdr_envmap_path)

    # Read list of envmap files and exposure infos
    envmap_files = glob.glob(os.path.join(envmap_dir, "*_er.tif"))
    with open(os.path.join(envmap_dir, EXPOSURE_INFO_FILENAME),
              "r", encoding="utf-8") as f:  # pylint: disable=redefined-outer-name
        exposure_infos = json.load(f)

    # height, width = 3648, 7296
    height, width = 1024, 2048
    img_sum = np.zeros(shape=(height, width, 3), dtype=float)
    weight_sum = np.zeros(shape=(height, width, 1), dtype=float)
    for envmap_file in envmap_files:
        # read image and exposure info
        filename = extract_filename(envmap_file)[:-3]  # remove "_er"
        exposure_info = exposure_infos[filename]
        ldr_image = tifffile.imread(envmap_file)
        ldr_image = to_float32(ldr_image)
        ldr_image = cv2.resize(ldr_image, dsize=(width, height),
                               interpolation=cv2.INTER_AREA)
        # ldr_image[-70:, :] = 0  # remove the camera pixels

        # compute weight
        brightness_image = ldr_image[..., 0] * 0.2126 + \
                           ldr_image[..., 1] * 0.7152 + \
                           ldr_image[..., 2] * 0.0722
        weight_image = np.exp(-4 * ((brightness_image - 0.5) ** 2) / (0.5 ** 2))
        weight_image = weight_image[:, :, None]

        # set the weight of pixels in extreme conditions to 0
        if envmap_file == envmap_files[-1]:  # is the brightest image
            weight_image[brightness_image < 0.2] = 1
        else:
            weight_image[brightness_image < 0.05] = 0
        if envmap_file == envmap_files[0]:  # is the darkest image
            weight_image[brightness_image > 0.7] = 1
        else:
            weight_image[brightness_image > 0.95] = 0

        img_sum += ldr_image / relative_exposure_amount(exposure_info) \
                   * weight_image
        weight_sum += weight_image

        write_image(os.path.join(save_dir, f"{filename}.png"),
                    ldr_image)
        write_image(os.path.join(save_dir, f"{filename}_w.png"),
                    weight_image)

    hdr_image = img_sum / (weight_sum + 1e-6) / \
                exposure_infos["over_exposure_coeff"]

    # save results
    write_image(hdr_envmap_path, hdr_image)

    return hdr_image


def get_brightness_image(img: np.ndarray) -> np.ndarray:
    """Get the brightness image of an image.

    Args:
        img (np.ndarray): the input image in linear sRGB space.

    Returns:
        np.ndarray: the brightness image.
    """
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


class SolidAngleCalculator:
    def __init__(self):
        pass

    def get_solid_angle(self, x, y):
        pass


class SolidAngleCalculatorForLimitedFOVImage(SolidAngleCalculator):
    """Calculate solid angle for a limited FOV image."""
    def __init__(self,
                 pixel_w: int, pixel_h: int,
                 sensor_w: float, sensor_h: float,
                 focal_length: float):
        """Initialize the solid angle calculator.

        Args:
            pixel_w (int): the height of the panorama image.
            pixel_h (int): the width of the panorama image.
        """
        self.pixel_w = pixel_w
        self.pixel_h = pixel_h
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h
        self.focal_length = focal_length

        self.pixel_area = self.sensor_w * self.sensor_h / \
                          (self.pixel_w * self.pixel_h)

    def get_solid_angle(self, x: int, y: int) -> float:
        """Get the solid angle of a pixel.

        Args:
            x (int): the horizontal coordinate of the pixel.
            y (int): the vertical coordinate of the pixel.

        Returns:
            float: the solid angle of the pixel.
        """
        # compute the displacement of the pixel from the center of the sensor
        x = (x + 0.5) / self.pixel_w
        y = (y + 0.5) / self.pixel_h
        x = x * self.sensor_w - self.sensor_w / 2
        y = y * self.sensor_h - self.sensor_h / 2
        # compute the distance r from the lens to the pixel.
        d = self.focal_length
        r = np.sqrt(x ** 2 + y ** 2 + d ** 2)
        # Solid angle = A * cos(theta) / r ** 2,
        # where theta is the angle between sensor normal and the ray from the
        # lens center to the pixel center, and thus cos(theta) = d / r.
        return self.pixel_area * d / (r ** 3)


class SolidAngleCalculatorForPanorama(SolidAngleCalculator):
    """Calculate solid angle for a panorama image."""
    def __init__(self, pixel_w: int, pixel_h: int):
        """Initialize the solid angle calculator.

        Args:
            pixel_w (int): the height of the panorama image.
            pixel_h (int): the width of the panorama image.
        """
        self.pixel_w = pixel_w
        self.pixel_h = pixel_h

    def get_solid_angle(self, x: int, y: int) -> float:
        """Get the solid angle of a pixel.

        Args:
            x (int): the horizontal coordinate of the pixel.
            y (int): the vertical coordinate of the pixel.

        Returns:
            float: the solid angle of the pixel.
        """
        elevation_angle = (y + 0.5) / self.pixel_h * np.pi - np.pi / 2
        return 2 * np.pi / self.pixel_w * \
               np.cos(elevation_angle) * np.pi / self.pixel_h


def compute_total_energy_and_solid_angle_above_threshold(
        hdr_img: np.ndarray,
        threshold: float,
        solid_angle_calculator: SolidAngleCalculator
) -> Tuple[np.ndarray, float, np.ndarray]:
    brightness_img = get_brightness_image(hdr_img)
    mask = brightness_img > threshold
    pixels_y, pixels_x = np.where(mask)
    total_solid_angle = 0.0
    total_energy = np.array([0, 0, 0], float)
    for py, px in tqdm(zip(pixels_y, pixels_x), total=len(pixels_x),
                       desc="Processing Pixels"):
        solid_angle = solid_angle_calculator.get_solid_angle(px, py)
        total_solid_angle += solid_angle
        total_energy += hdr_img[py, px, :] * solid_angle
    return total_energy, total_solid_angle, mask


def check_bright_part_of_hdr_image(
        hdr_img: np.ndarray,
        img_name: str,
        bright_threshold: float,
        solid_angle_calculator: SolidAngleCalculator,
        check_full_image: bool = False,):
    if check_full_image:
        # print min, max and average of brightness
        brt_img = get_brightness_image(hdr_img)
        print(f"{img_name} brightness: "
              f"min = {np.min(brt_img)}, "
              f"max = {np.max(brt_img)}, "
              f"average = {np.mean(brt_img)}")

        # compute total energy and solid angle in the image
        total_energy, total_solid_angle, _ = \
            compute_total_energy_and_solid_angle_above_threshold(
                hdr_img, 0, solid_angle_calculator)
        print(f"{img_name} total energy: {total_energy}")
        print(f"{img_name} total solid angle: {total_solid_angle}")
        print(f"{img_name} total avg. intensity: "
              f"{total_energy / total_solid_angle}")

    # Compute total energy and solid angle in the bright part of the image
    total_bright_energy, total_bright_solid_angle, bright_mask = \
        compute_total_energy_and_solid_angle_above_threshold(
            hdr_img, bright_threshold, solid_angle_calculator)
    print(f"{img_name} bright area energy: {total_bright_energy}")
    print(f"{img_name} bright area solid angle: {total_bright_solid_angle}")
    print(f"{img_name} bright area avg. intensity: "
          f"{total_bright_energy / total_bright_solid_angle}")
    # write_image(os.path.join(save_dir, f"{img_name}_bright_mask.png"),
    #             bright_mask)

    return total_bright_energy, total_bright_solid_angle, bright_mask


if __name__ == "__main__":
    # TODO: automate this script to find all envmap directories and process
    #  them one by one.

    # ===== Step 1: Calibrate Color Correction Matrix =====
    print("===== Step 1: Calibrate Color Correction Matrix =====")
    # Raw file paths
    sony_raw_file = os.path.join(CALIB_DATA_DIR, "sony.ARW")
    ricoh_raw_file = os.path.join(CALIB_DATA_DIR, "ricoh.DNG")
    sony_raw_file_w_filter = os.path.join(CALIB_DATA_DIR, "sony_wND.ARW")
    sony_raw_file_wo_filter = os.path.join(CALIB_DATA_DIR, "sony_woND.ARW")
    # Calibrate the CCM from Ricoh to Sony
    ccm_ricoh2sony, over_exposure_ricoh2sony = calibrate_ccm_ricoh2sony(
        sony_raw_file,
        ricoh_raw_file)
    density_ndfilter, ccm_ndfilter = calibrate_ndfilter(
        sony_raw_file_w_filter,
        sony_raw_file_wo_filter)
    print("===== Step 1: Done =====\n")

    # ===== Step 2: Preprocess a group of raw envmap images =====
    print("===== Step 2: Preprocess a group of raw envmap images =====")
    group_path = os.path.abspath("./Data/0926_sun_2/")
    preprocess_envmap_group(os.path.join(group_path, "ricoh"),
                            ccm_ricoh2sony,
                            over_exposure_ricoh2sony)
    print("===== Step 2: Done =====\n")

    # ===== Step 3: Manually stitch envmap images into a panorama =====
    # In this step you need to manually stitching the envmaps using RICOH THETA
    # STITCHER. Remember to calibrate the pitch and row so that Sony camera is
    # at the center position. Save the stitched image as a .TIF file with suffix
    # "_er.TIF" as the default setting of RICOH THETA STITCHER.
    print("===== Step 3: Manually stitch envmap images into a panorama =====")
    assert ensure_envmaps_are_stitched(os.path.join(group_path, "ricoh")), \
        "Please stitch the envmaps using RICOH THETA STITCHER first!"
    print("The LDR envmaps have been stitched into HDR.")
    print("===== Step 3: Done =====\n")

    # ===== Step 4: Combine the LDR panoramas into a HDR panorama =====
    print("===== Step 4: Combine the LDR panoramas into a HDR panorama =====")
    envmap_hdr = merge_ldrs_into_hdr(os.path.join(group_path, "ricoh"))

    solid_angle_calculator_envmap = \
        SolidAngleCalculatorForPanorama(envmap_hdr.shape[1],
                                        envmap_hdr.shape[0])

    envmap_bright_energy, envmap_bright_solid_angle, envmap_bright_mask = \
        check_bright_part_of_hdr_image(
            envmap_hdr,
            "envmap",
            get_brightness_image(envmap_hdr).max() * 0.95,
            solid_angle_calculator_envmap,
            check_full_image=True)
    print("===== Step 4: Done =====\n")

    # ===== Step 5: Correct the sun intensity =====
    # In the case that the panorama includes the sun, its intensity is often
    # clipped. We need to compute the unclipped intensity of the sun from the
    # image taken with the ND filter.

    # Post-process the image taken with the ND filter, correcting its color
    # shift caused by the ND filter.
    print("===== Step 5: Correct the sun intensity =====")
    sun_image_file = os.path.join(group_path, "sony", "CI_01838.ARW")
    sun_image_save_file = os.path.join(group_path, "sony", "CI_01838.png")
    if not os.path.exists(sun_image_save_file):
        print(f"Processing sun image {sun_image_file}...")
        sun_image = camera_isp(sun_image_file,
                               ccm_ndfilter,
                               MY_DEFAULT_WB,
                               SONY_CAM2XYZ_MAT)
        write_image(sun_image_save_file, sun_image)
    else:
        print(f"Loading sun image {sun_image_save_file}...")
        sun_image = read_image(sun_image_save_file)

    # Correct the intensity by considering the ND filter density and the
    # relative exposure amount.
    sun_image_exposure_info = read_exposure_from_raw(sun_image_file)
    sun_image_relative_exposure = \
        relative_exposure_amount(sun_image_exposure_info)
    print(f"Sun image magnification: {1 / sun_image_relative_exposure}\n")
    sun_image = sun_image * density_ndfilter / sun_image_relative_exposure

    # Find the sun position in the image by thresholding the brightness image.
    sun_brt_img = get_brightness_image(sun_image)
    sun_position = np.unravel_index(np.argmax(sun_brt_img),
                                    sun_brt_img.shape)
    sun_intensity = sun_image[sun_position[0], sun_position[1], :]
    print(f"Sun intensity: {sun_intensity}")

    print("===== Step 5: Done =====\n")

    # ===== Step 6: Migrate the sun intensity to the HDR envmap  =====
    print("===== Step 6: Migrate the sun intensity to the HDR envmap =====")
    # Step 6.1: Compute the total energy and solid angle of the sun in the
    # ND-filtered image (sun image).

    # read the focal length from the EXIF data
    with open(sun_image_file, "rb") as f:
        exif_info = exifread.process_file(f, details=False)
        sun_img_focal_length = float(exif_info["EXIF FocalLength"].values[0])
    # Compute the solid angle of each pixel
    solid_angle_calculator_sony = SolidAngleCalculatorForLimitedFOVImage(
        SONY_CAM_PIXEL_W, SONY_CAM_PIXEL_H,
        SONY_CAM_SENSOR_W, SONY_CAM_SENSOR_H,
        sun_img_focal_length
    )
    sun_bright_energy, sun_bright_solid_angle, sun_bright_mask = \
        check_bright_part_of_hdr_image(
            sun_image,
            "sun image",
            get_brightness_image(sun_image).max() * 0.1,
            solid_angle_calculator_sony,
            check_full_image=False)

    # Step 6.2: Migrate the sun intensity to the HDR envmap
    energy_difference = sun_bright_energy - envmap_bright_energy
    mean_intensity = energy_difference / envmap_bright_solid_angle
    print(f"Mean intensity to be added to the envmap: {mean_intensity}")
    envmap_hdr_sun = envmap_hdr + \
                     mean_intensity.reshape(1, 1, 3) * \
                     envmap_bright_mask.reshape(
                         envmap_hdr.shape[0], envmap_hdr.shape[1], 1)
    write_image(os.path.join(group_path, "envmap", "envmap_with_sun.hdr"),
                             envmap_hdr_sun)
    print("===== Step 6: Done =====\n")




