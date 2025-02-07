# AKAZE with SVM Classifier

## Overview

This repository contains an implementation of an **image classification pipeline** using the **AKAZE (Accelerated KAZE) feature detector** combined with an **SVM (Support Vector Machine) classifier**. The project aims to detect and classify objects in images using robust and rotation-invariant feature extraction.

## Why AKAZE?

✅ If you need to detect rotating objects (e.g., wind turbines).

✅ If you want to use a fast and free algorithm.

✅ If you are working with images at different scales.

## Features

- **AKAZE Feature Extraction**: Detects keypoints and extracts descriptors that are invariant to scale and rotation.
- **SVM Classification**: Uses a Support Vector Machine to classify objects based on extracted AKAZE features.
- **OpenCV Integration**: Uses OpenCV for feature extraction, image processing and SVM classification.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/Mac-FF/AKAZE-with-SVM.git
cd AKAZE-with-SVM-Classifier

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.x
- OpenCV 4.11.0.86
- NumPy 2.2.2

## Usage

### Extract Features and Train SVM

Run the script to extract AKAZE features and train the SVM classifier:

```bash
python train.py <path to training positive dataset> <path to training negative dataset> <output path for results>
```

### Perform Classification on dataset

Use the trained SVM model to classify images in dir:

```bash
python classify.py <path to svm model> <path to dataset>
```

## Dataset Format

Scripts support PNG and JPG images. The dataset should be structured as follows:

```
dataset/
│── <positive>/
│   ├── image1.jpg
│   ├── image2.jpg
│── <negative>/
│   ├── image1.jpg
│   ├── image2.jpg
...
```

## Results

The results are saved in yml and json files.
