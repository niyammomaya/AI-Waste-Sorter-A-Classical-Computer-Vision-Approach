import argparse
import os
import cv2
import numpy as np
from features import extract_features_from_item
from utils import load_images_from_dir, simple_foreground_mask
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # progress bar

def main(args):
    classes = args.classes
    all_features = []
    all_labels = []

    print("\n===============================")
    print("     AI WASTE SORTER TRAIN     ")
    print("===============================\n")

    print("[INFO] Loading and processing dataset...\n")

    # Loop over each class
    for label, cls_name in enumerate(classes):
        folder = os.path.join(args.data_dir, cls_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder for class '{cls_name}'")
            continue

        print(f"[CLASS] {cls_name} — extracting features...")

        paths = load_images_from_dir(folder)

        # tqdm progress bar
        for p in tqdm(paths, desc=f"{cls_name:>10s}", ncols=80):
            img = cv2.imread(p)
            if img is None:
                continue

            mask = simple_foreground_mask(img, invert_if_needed=True)
            feats, _ = extract_features_from_item(img, mask)

            if feats is not None:
                all_features.append(feats)
                all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    print("\n[INFO] Scaling features...")
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    os.makedirs(args.models_dir, exist_ok=True)

    print("[INFO] Training SVM classifier...\n")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.5)
    svm.setGamma(0.02)

    svm.train(
        all_features.astype(np.float32),
        cv2.ml.ROW_SAMPLE,
        all_labels.astype(np.int32)
    )

    print("[INFO] Saving model + scaler...")

    # Save model
    svm.save(os.path.join(args.models_dir, "waste_svm_model.yml"))

    # Save scaler
    fs = cv2.FileStorage(os.path.join(args.models_dir, "scaler.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("mean", scaler.mean_)
    fs.write("std", scaler.scale_)
    fs.write("classes", classes)
    fs.release()

    print("\n[OK] Training complete!")
    print("[OK] Model saved to models/waste_svm_model.yml")
    print("[OK] Scaler saved to models/scaler.yml\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()
    main(args)