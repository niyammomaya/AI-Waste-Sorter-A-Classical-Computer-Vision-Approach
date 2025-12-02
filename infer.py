import argparse
import os
import cv2
import glob
import numpy as np
from utils import simple_foreground_mask
from features import extract_features_from_item

def load_models(models_dir):
    # Load scaler
    fs = cv2.FileStorage(os.path.join(models_dir, "scaler.yml"), cv2.FILE_STORAGE_READ)
    mean = fs.getNode("mean").mat().flatten()
    std = fs.getNode("std").mat().flatten()

    cls_node = fs.getNode("classes")
    classes = [cls_node.at(i).string() for i in range(cls_node.size())]
    fs.release()

    # Load SVM
    svm = cv2.ml.SVM_load(os.path.join(models_dir, "waste_svm_model.yml"))

    return mean, std, classes, svm

def main(args):
    mean, std, classes, svm = load_models(args.models_dir)

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.images_dir, e)))

    if not paths:
        print("[ERR] No test images found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for p in paths:
        fname = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue

        # Mask + features
        mask = simple_foreground_mask(img, invert_if_needed=True)
        feats, _ = extract_features_from_item(img, mask)
        if feats is None:
            continue

        # Normalize
        feats = (feats - mean) / std

        # Predict
        _, pred = svm.predict(feats.reshape(1, -1).astype(np.float32))
        pred_idx = int(pred[0, 0])
        pred_class = classes[pred_idx]

        print(f"[OK] {fname} → {pred_class}")

        # Save the output image (unchanged)
        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    main(args)