"""
Codes for reading exposure information from raw images and computing the
relative exposure amount with respect to the standard exposure setting.
"""
from typing import Dict

import exifread


BASE_EXPOSURE = {
    "ISO": 100.0,
    "shutter speed": 0.01,
    "aperture": 3.5,
}


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
