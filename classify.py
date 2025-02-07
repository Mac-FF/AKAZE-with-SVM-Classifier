import os
import argparse
import cv2 as cv
import numpy as np
import sample as sp
import graphicshelpers as gh
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Use the trained SVM model to classify images in dir')
    parser.add_argument('model_svm_name', type=gh.is_file,
                        help='path to svm model')

    parser.add_argument('data_path', type=gh.is_dir,
                        help='path to dataset')
    return parser.parse_args()

def main():
    # Classification
    data_time = gh.print_version()
    parsed_args = parse_arguments()

    model_svm_file_name = Path(parsed_args.model_svm_name)
    testing_data_path = Path(parsed_args.data_path)
    results_dir = os.path.dirname(str(model_svm_file_name))
    debug_dir = results_dir + os.sep + 'classify_debug' + os.sep
    gh.mkdir(debug_dir)

    json_data = {
        'datatime': data_time,
        'revision': '',
        'build date': '',
        'output path': results_dir,
        'testing dataset': str(testing_data_path),
        'testing model': str(model_svm_file_name),
        'n object detected': 0,
        'object detected': [],
        'n no object': 0,
        'no object': []
    }

    # Read model
    print('Model:', model_svm_file_name)
    svm = cv.ml.SVM_load(model_svm_file_name)

    testing_files = gh.filter_files(testing_data_path)
    for index, test_file in enumerate(testing_files):
        n_images = index + 1
        test_file_name = str(test_file)
        print(n_images, ": ", test_file_name)
        out_debug_name = debug_dir + str(n_images) + '_' + os.path.basename(test_file_name)[:-4]
        label = -1
        sample = sp.Sample(test_file_name, label)
        gh.save_image(out_debug_name, sample.get_image(), 'image')
        gh.save_image(out_debug_name, sample.get_image_filtered(), 'image_filtered')
        gh.save_image(out_debug_name, sample.get_image_kp(), 'image_kp')

        desc = sample.get_desc()
        if desc is not None:
            results = []
            for des in desc:
                d  = des.reshape(1, -1).astype("float32")
                result = svm.predict(d)[1].ravel()
                results.append(result[0])

            mean_prediction = float(np.mean(results))
            prediction = 1 if mean_prediction > 0.5 else 0
            if prediction == 1:
                print('Object detected')
                json_data["object detected"].append([mean_prediction, test_file_name])
            else:
                print('No object')
                json_data["no object"].append([mean_prediction, test_file_name])
        else:
            print('No features in the test image')

    json_data["n object detected"] = len(json_data["object detected"])
    json_data["n no object"] = len(json_data["no object"])

    # Save testing results
    results_file_name = results_dir + os.sep + 'test_results.json'
    print(json_data)
    print('test results:', results_file_name)
    gh.write_json(results_file_name, json_data)

if __name__ == "__main__":
    main()