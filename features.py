import cv2
import numpy as np

def hsv_hist(bgr, mask=None, h_bins=16, s_bins=8, v_bins=8):
    """Compute a normalized HSV histogram (flattened)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], mask, [h_bins, s_bins, v_bins], [0,180, 0,256, 0,256])
    hist = cv2.normalize(hist, None).flatten()
    return hist.astype(np.float32)

def lbp_hist(gray, P=8, R=1):
    """Simple LBP (non-uniform). Returns a 256-bin normalized histogram.

    Note: For speed and simplicity, this uses nearest-neighbor sampling and raw 8-bit codes.
    """
    h, w = gray.shape
    dst = np.zeros((h-2*R, w-2*R), dtype=np.uint8)
    for y in range(R, h-R):
        for x in range(R, w-R):
            c = gray[y, x]
            code = 0
            for p in range(P):
                theta = 2*np.pi*p/P
                yy = int(round(y + R*np.sin(theta)))
                xx = int(round(x + R*np.cos(theta)))
                code = (code << 1) | (1 if gray[yy, xx] >= c else 0)
            dst[y-R, x-R] = code
    hist = cv2.calcHist([dst], [0], None, [256], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return hist.astype(np.float32)

def shape_features(cnt):
    """Basic shape descriptors + Hu moments."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = (area / (rect_area + 1e-6))
    aspect = (w / (h + 1e-6))
    compact = (4*np.pi*area) / ((peri + 1e-6) ** 2)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    feats = np.array([area, peri, extent, aspect, compact, *hu], dtype=np.float32)
    return feats

def extract_features_from_item(bgr, mask):
    """Given a BGR image and its binary foreground mask, compute feature vector."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None, None
    x,y,w,h = cv2.boundingRect(c)
    crop = bgr[y:y+h, x:x+w]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_mask = mask[y:y+h, x:x+w]

    f_color = hsv_hist(crop, mask=crop_mask)
    f_shape = shape_features(c)
    f_lbp   = lbp_hist(crop_gray)

    feats = np.concatenate([f_color, f_shape, f_lbp]).astype(np.float32)
    bbox = (x,y,w,h)
    return feats, bbox
