# Brain MRI Tumor Classification using CNN with Attention and Image Preprocessing

## Overview

This project uses a Convolutional Neural Network (CNN) combined with a Convolutional Block Attention Module (CBAM) to classify brain MRI images into four different tumor types. It compares the performance of the model using various image preprocessing techniques, including Contrast Limited Adaptive Histogram Equalization (CLAHE).

The four distinct tumor classes are:
* `glioma`
* `meningioma`
* `notumor` (no tumor)
* `pituitary`

## Dependencies

The notebook uses the following major libraries:

* `tensorflow` (and `tensorflow.keras`)
* `pandas`
* `numpy`
* `opencv-python` (`cv2`)
* `scikit-image` (`skimage.feature`, `skimage.measure`)
* `matplotlib`
* `seaborn`
* `scikit-learn` (`sklearn.svm`, `sklearn.ensemble`, `sklearn.preprocessing`, `sklearn.metrics`)

## Dataset

The notebook is configured to load and extract data from a file named `dataset.zip`, which is expected to contain `Training` and `Testing` directories.

* **Training Images:** 5712
* **Testing Images:** 1311

## Preprocessing & Feature Extraction

The code implements a loading function that resizes images to `(240, 240)`.

It also defines a `apply_clahe` function, which performs **Contrast Limited Adaptive Histogram Equalization (CLAHE)**, an image enhancement technique, on the L channel of the LAB color space.

The notebook likely uses additional feature extraction, such as **Gray-Level Co-occurrence Matrix (GLCM)** properties, as indicated by the imports from `skimage.feature.graycomatrix` and `skimage.feature.graycoprops`.

## Model Architecture

The model is a CNN that incorporates a **Convolutional Block Attention Module (CBAM)**, implemented using custom Keras layers:

1.  **`ChannelAttention` Layer:** Uses global average pooling and global max pooling, followed by a Multi-Layer Perceptron (MLP) and a sigmoid activation to recalibrate channel-wise feature responses.
2.  **`SpatialAttention` Layer:** Uses 2D convolution and sigmoid activation on concatenated average-pooled and max-pooled feature maps across the channel dimension to generate spatial attention maps.
3.  **`CBAM` Layer:** Combines the `ChannelAttention` and `SpatialAttention` mechanisms.

## Results Comparison

The notebook evaluates the model's performance across three versions, primarily differing by the image processing applied to the input data:

| Version | Preprocessing/Features | Validation Accuracy (Test Set) |
| :--- | :--- | :--- |
| **Version 1** | No Preprocessing | 98.245614% |
| **Version 2** | CLAHE (Contrast Enhancement) | 98.703280% |
| **Version 3** | Preprocessing (e.g., CLAHE + GLCM/Texture Features) | 98.932113% |
