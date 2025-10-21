"""
Problem Statement (Frequency Domain Filtering):
Apply Butterworth, Gaussian, and Ideal filtering (low-pass, high-pass, band-pass)
on grayscale images with low, normal, and high contrast.
Visualize the filtered outputs for each contrast level and filter type.
"""

#================= Importing necessary libraries ======================
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#================= Output Directory ==================================
output_dir = "./hw_15_output"
os.makedirs(output_dir, exist_ok=True)
img_counter = 0

#================= Main Execution Workflow ============================
def main():
    # Load original grayscale image
    img = cv2.imread('/home/kowsar/Documents/Digital_Image_Processing/DIP_Problems/Images/tulip.png', 0)

    # Create contrast variants
    low_img = cv2.convertScaleAbs(img, alpha=0.5, beta=40)   # Low contrast
    norm_img = img.copy()                                    # Normal contrast
    high_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)   # High contrast

    # Apply filtering on contrast variants
    filtering(low_img, norm_img, high_img)

    # Compare filter types on normal contrast
    compare_filters(norm_img)

    # Vary Butterworth filter order
    vary_butterworth_order(norm_img)


#================= Part 1: Filtering on Contrast Variants =============
def filtering(low_img, norm_img, high_img):
    for img, label in zip([low_img, norm_img, high_img], ["Low", "Normal", "High"]):
        img_set_gaussian = [img]
        img_set_butter = [img]
        img_title_gaussian = [f"{label}-Contrast"]
        img_title_butter = [f"{label}-Contrast"]

        for ftype in ["low", "high", "band"]:
            H_butter = butterworth_filter(img.shape, type=ftype, n=2)
            H_gauss = gaussian_filter(img.shape, type=ftype)

            img_set_butter.append(apply_filter(img, H_butter))
            img_title_butter.append(f"Butterworth-{ftype}")

            img_set_gaussian.append(apply_filter(img, H_gauss))
            img_title_gaussian.append(f"Gaussian-{ftype}")

        display(img_set_gaussian, img_title_gaussian)
        display(img_set_butter, img_title_butter)


#================= Part 2: Comparing Filter Types =====================
def compare_filters(img_gray):
    img_set = []
    img_title = []

    for ftype in ["low", "high", "band"]:
        H_butter = butterworth_filter(img_gray.shape, type=ftype, n=2)
        H_gauss = gaussian_filter(img_gray.shape, type=ftype)
        H_ideal = ideal_filter(img_gray.shape, type=ftype)

        img_set.append(apply_filter(img_gray, H_butter))
        img_title.append(f"Butterworth-{ftype}")

        img_set.append(apply_filter(img_gray, H_gauss))
        img_title.append(f"Gaussian-{ftype}")

        img_set.append(apply_filter(img_gray, H_ideal))
        img_title.append(f"Ideal-{ftype}")

    display_2(img_set, img_title)


#================= Part 3: Varying Butterworth Order ==================
def vary_butterworth_order(img_gray):
    img_set = []
    img_title = []

    for n in [1, 2, 5, 10]:
        H = butterworth_filter(img_gray.shape, type='low', n=n)
        filtered = apply_filter(img_gray, H)
        img_set.append(filtered)
        img_title.append(f"Butterworth Low-pass (n={n})")

    display(img_set, img_title)


#================= Butterworth Filter ================================
def butterworth_filter(shape, type='low', n=2):
    # Ensure low < high for correct band-pass
    radius_low = shape[0] // 6
    radius_high = shape[0] // 4

    if type == 'low':
        return butterworth_low_pass_filter(shape, cutoff=radius_low, n=n)
    elif type == 'high':
        return 1 - butterworth_low_pass_filter(shape, cutoff=radius_low, n=n)
    elif type == 'band':
        return butterworth_low_pass_filter(shape, cutoff=radius_high, n=n) - \
               butterworth_low_pass_filter(shape, cutoff=radius_low, n=n)


def butterworth_low_pass_filter(shape, cutoff, n):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U*2 + V*2)
    epsilon = 1e-6  # prevent division by zero
    D_safe = D + epsilon
    return 1 / (1 + (D_safe / cutoff)**(2 * n))


#================= Gaussian Filter ===================================
def gaussian_filter(shape, type='low'):
    radius_low = shape[0] // 6
    radius_high = shape[0] // 4

    if type == 'low':
        return gaussian_low_pass_filter(shape, cutoff=radius_low)
    elif type == 'high':
        return 1 - gaussian_low_pass_filter(shape, cutoff=radius_low)
    elif type == 'band':
        return gaussian_low_pass_filter(shape, cutoff=radius_high) - \
               gaussian_low_pass_filter(shape, cutoff=radius_low)


def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U*2 + V*2)
