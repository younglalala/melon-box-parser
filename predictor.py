# coding=utf-8
import os

import cv2
from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model(r"./box-parser-tensorflow/models/1524290339")

total_count = 0
error_count = 0

base_dir = r"./box-parser-data-set"
for filename in os.listdir(base_dir):
    filepath = os.path.join(base_dir, filename)
    image = cv2.imread(filepath, 0)
    predict_result = predict_fn(dict(x=image.reshape(18, 66)))

    real_option = filename[-5]
    predict_option = chr(predict_result['classes'][0] + ord('A'))

    total_count += 1

    if predict_result['probabilities'][0][predict_result['classes'][0]] <= 0.7:
        continue

    if real_option != predict_option:
        error_count += 1
        print("real_option {} predict_option {} filename {}".format(real_option, predict_option, filename))
        print("total_count {} error_count {} percent {}%".format(total_count, error_count, error_count * 100 / total_count))
