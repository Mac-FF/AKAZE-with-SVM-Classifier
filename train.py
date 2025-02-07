import os
import json
import argparse
import cv2 as cv
import numpy as np
import sample as sp
import graphicshelpers as gh
from pathlib import Path
#import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the script to extract AKAZE features and train the SVM classifier')
    parser.add_argument('pos_data_path', type=gh.is_dir,
                        help='path to positive dataset with images')

    parser.add_argument('neg_data_path', type=gh.is_dir,
                        help='path to negative dataset with images')

    parser.add_argument('output_path', type=gh.is_results_dir,
                        help='output path for results')
    # parser.add_argument('--classify', action='store_true')
    # parser.add_argument('--no-classify', dest='classify', action='store_false')
    # parser.set_defaults(classify=True)
    return parser.parse_args()

# def run_classify(path):
#     subprocess.run(['python', path])


def main():
    # Training
    data_time = gh.print_version()
    # Parse arguments
    parsed_args = parse_arguments()
    pos_data_path = Path(parsed_args.pos_data_path)
    neg_data_path = Path(parsed_args.neg_data_path)
    results_dir = str(Path(parsed_args.output_path)) + os.sep + data_time
    gh.mkdir(results_dir)
    debug_dir = results_dir + os.sep + 'train_debug' + os.sep
    gh.mkdir(debug_dir)
    model_svm_file_name = str(results_dir) + os.sep + 'model_svm.yml'

    pos_files = gh.filter_files(pos_data_path)
    neg_files = gh.filter_files(neg_data_path)
    pos_labels = [1] * len(pos_files)
    neg_labels = [0] * len(neg_files)
    labels = pos_labels + neg_labels
    files = pos_files + neg_files

    json_data = {
        'datatime': data_time,
        'revision': '',
        'build date': '',
        'positive dataset': str(pos_data_path),
        'negative dataset': str(neg_data_path),
        'output path': results_dir,
        'output model': model_svm_file_name,
        'n training images': 0,
        'training images': [],
        'n missed images': 0,
        'missed images': []
    }
    features = []
    for index, file in enumerate(files):
        n_images = index + 1
        file_name = str(file)
        print(n_images, ": ", file_name)
        label = labels[index]
        out_debug_name = debug_dir + str(n_images) + '_' + os.path.basename(file_name)[:-4]

        sample = sp.Sample(file_name, label)
        gh.save_image(out_debug_name, sample.get_image(), 'image')
        gh.save_image(out_debug_name, sample.get_image_filtered(), 'image_filtered')
        gh.save_image(out_debug_name, sample.get_image_kp(), 'image_kp')

        # Prepare training data
        desc = sample.get_desc()
        if desc is not None:
            json_data["training images"].append(file_name)
            desc = desc.astype("float32")
            for d in desc:
                features.append((d, label))
        else:
            json_data["missed images"].append(file_name)

    train_data = np.array([f[0] for f in features])
    train_labels = np.array([f[1] for f in features])

    json_data["n training images"] = len(json_data["training images"])
    json_data["n missed images"] = len(json_data["missed images"])

    # SVM params
    print('Training...')
    svm = cv.ml.SVM_create()
    svm_kernel = cv.ml.SVM_LINEAR # cv.ml.SVM_CHI2
    svm_type = cv.ml.SVM_C_SVC
    svm_c = 2.67
    svm_gamma = 5.383
    svm.setKernel(svm_kernel)
    svm.setType(svm_type)
    svm.setC(svm_c)
    svm.setGamma(svm_gamma)
    svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)

    # Save SVM model
    print('Model saved to:', model_svm_file_name)
    svm.save(model_svm_file_name)

    results_file_name = model_svm_file_name[:-4] + '.json'
    print('Training results:', results_file_name)

    gh.write_json(results_file_name, json_data)


if __name__ == "__main__":
    main()


