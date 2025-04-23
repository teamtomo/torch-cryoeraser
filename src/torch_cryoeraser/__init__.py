"""Erase masked regions from cryo-EM images"""

__version__ = '0.1.0'
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .erase import erase_region_2d

__all__ = [
    "erase_region_2d"
]