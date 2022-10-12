"""
Codes for checking the statistics of the environment map.
"""
from typing import Tuple

import numpy as np
from tqdm import tqdm

from solid_angle import SolidAngleCalculator
from utils import get_brightness_image


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
        check_full_image: bool = False, ):
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
