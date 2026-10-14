"""Spectral indices for burned-area detection."""
import numpy as np


# CaBuAr 12-band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
B02_BLUE = 1
B03_GREEN = 2
B04_RED = 3
B08_NIR = 7
B11_SWIR1 = 10
B12_SWIR2 = 11


def compute_nbr(image):
    """
    Normalized Burn Ratio (NBR).
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    image: (12, H, W) float
    returns: (H, W) in [-1, 1]
    """
    nir = image[B08_NIR].astype(np.float32)
    swir2 = image[B12_SWIR2].astype(np.float32)
    return (nir - swir2) / (nir + swir2 + 1e-8)


def compute_dnbr(pre_image, post_image):
    """
    Differenced NBR (dNBR).
    dNBR = NBR_pre - NBR_post
    Positive values indicate vegetation loss (burning).
    """
    return compute_nbr(pre_image) - compute_nbr(post_image)


def compute_bai(image):
    """
    Burn Area Index (BAI).
    BAI = 1 / ((0.1 - Red)^2 + (0.06 - NIR)^2)
    Higher values indicate burned area.
    """
    red = image[B04_RED].astype(np.float32) / 255.0
    nir = image[B08_NIR].astype(np.float32) / 255.0
    return 1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + 1e-8)
