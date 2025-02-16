import os
import sys
import math
import json
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime


_debug_level = 3

def is_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid path")

def is_dir(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def is_results_dir(path):
    if os.path.isdir(path):
        return path
    else:
        try:
            os.mkdir(path)
            return path
        except OSError as error:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path: {error}")

def mkdir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

def write_json(file_name, data):
    with open(file_name,'w+') as file:
        file.seek(0)
        json.dump(data, file, indent=4, sort_keys=False)

def print_version():
    system = os.name
    data_time = datetime.now()
    data_time = data_time.strftime("%Y-%m-%d_%H.%M.%S")
    print("Time:", data_time)
    print("System:", system)
    print("Python ver.:", sys.version)
    print("OpenCV ver.:", cv.__version__)
    print("NumPy ver.:", np.__version__)
    return data_time

def filter_files(dir_path):
    return [filename for filename in dir_path.glob('**/*.*') if str(filename).endswith(('jpg', 'png'))]

def read_image(file_name):
    image = cv.imread(file_name, cv.IMREAD_GRAYSCALE | cv.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        print(f"Error: Could not read image {file_name}")
    return image

def save_image(file_name, image, comment):
    if _debug_level >= 3:
        out_name = file_name  + '_' + comment + '.png'
        cv.imwrite(out_name, image)

def prepare_image(image):
    image = scale_up(image)
    image = blur(image)
    return equalize_brightness(image)

def scale_up(image):
    return cv.resize(image, dsize=None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)

def blur(image):
    sigma = 5.0
    return cv.bilateralFilter(image, -1, sigma, sigma)

def equalize_brightness(image):
    ref_brightness = 128
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    sum = image.shape[0] * image.shape[1]
    total = 0
    median = 0
    for i in range(256):
        total += hist[i]
        if total >= sum / 2:
            median = i
            break
    gamma = math.log(ref_brightness / 255) / math.log(median / 255)
    lookUpTable = np.zeros((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    image = cv.LUT(image, lookUpTable)
    return image

def extract_fetures(image):
    akaze_thresh = 0.00001
    min_points = 4
    max_points = 16
    akaze = cv.AKAZE_create(threshold=akaze_thresh, max_points=max_points)
    kp, desc = akaze.detectAndCompute(image, None)
    if len(kp) < min_points:
        desc = None
        kp = None
    return kp, desc