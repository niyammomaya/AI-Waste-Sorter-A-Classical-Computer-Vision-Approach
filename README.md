# AI-Waste-Sorter-A-Classical-Computer-Vision-Approach
This project demonstrates that a system built on classical computer vision and machine learning principles can effectively classify common recyclables in real time.

AI Waste Sorter — OpenCV-Only (No Hardware)

A simple, explainable **waste classification** project using **pure OpenCV** + classic features
(Color histograms, Shape descriptors, and LBP texture) and an **OpenCV SVM** classifier.

> Focus classes for the MVP: `paper`, `plastic`, `metal`. (You can add more later.)

## Folder Structure
```
ai-waste-sorter-opencv/
├─ dataset/
│  ├─ paper/      # your images (jpg/png)
│  ├─ plastic/
│  └─ metal/
├─ models/
├─ features.py
├─ utils.py
├─ train.py
├─ infer.py
├─ requirements.txt
└─ README.md
```

## Quickstart
1. **Install deps** (Python 3.9+ recommended)
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Put your images in `dataset/<class>/` folders (e.g., `dataset/paper/xxx.jpg`).
   - Recommend ~100–300 images per class to start.
   - Use a fairly uniform background for easier segmentation.

3. **Train**
   ```bash
   python train.py --data_dir dataset --classes paper plastic metal --models_dir models
   ```

4. **Infer (webcam)**
   ```bash
   python infer.py --models_dir models --classes paper plastic metal --camera 0
   ```

5. **Infer (on a folder of images)**
   ```bash
   python infer.py --models_dir models --classes paper plastic metal --images_dir path/to/images
   ```

## Notes
- This is an **OpenCV-first** solution: no heavy deep learning required.
- You can add classes by collecting images and retraining with the `--classes` list.
- If segmentation is noisy, try:
  - Better lighting / plain background
  - Adjust kernel sizes (`--open_k`, `--close_k`) and thresholds (`--diff_thresh`).
- For best results, capture data with the **same camera & setup** used during inference.
