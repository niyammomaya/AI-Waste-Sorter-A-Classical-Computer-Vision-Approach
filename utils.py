import cv2
import numpy as np
import os, glob

def load_images_from_dir(dir_path, exts=(".jpg",".jpeg",".png",".bmp")):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(dir_path, f"*{e}")))
    return paths

def simple_foreground_mask(img, invert_if_needed=True):
    """Create a simple foreground mask assuming a relatively uniform background.

    Uses Otsu threshold on blurred grayscale. If the foreground is brighter than
    background the mask is fine; if it's darker, we may invert based on area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_if_needed:
        # Decide whether to invert based on which side has larger area near borders
        h, w = th.shape
        border = np.zeros_like(th); border[:5,:]=255; border[-5:,:]=255; border[:,:5]=255; border[:,-5:]=255
        white_border = cv2.countNonZero(cv2.bitwise_and(th, border))
        black_border = cv2.countNonZero(cv2.bitwise_and(255-th, border))
        if white_border > black_border:
            th = 255 - th
    # Clean up
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return th

def running_avg_bg_mask(gray, bg, alpha=0.01, diff_thresh=25, open_k=5, close_k=9):
    """Update running average background and produce a binary mask of moving/placed objects."""
    cv2.accumulateWeighted(gray, bg, alpha)
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg))
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if open_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k,open_k), np.uint8))
    if close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k,close_k), np.uint8))
    return mask
