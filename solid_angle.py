"""
Tool for computing solid angles for every pixel in an image.
"""
import numpy as np


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
