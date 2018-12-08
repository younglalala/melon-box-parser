# coding=utf-8
import os
import random

import cv2
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # image 18 x 66 x 1
    input_layer = tf.reshape(features["x"], [-1, 18, 66, 1])

    # 18 x 66 x 1 => 18 * 66 * 32
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
    )

    # 18 x 66 x 32 => 9 x 33 x 32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

    # 9 x 33 x 32 => 9 x 33 x 64
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
    )

    # 9 x 33 x 64 => 3 x 11 x 64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(3, 3), strides=3)

    # 3 x 11 x 64 => 2112
    pool2_flat = tf.reshape(pool2, [-1, 3 * 11 * 64])

    # 16384 => 1024
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # 1024
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 10 -> 3
    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "prediction": tf.estimator.export.PredictOutput(predictions),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def data_input_fn(mode):
    base_path = r"./data-set"

    filenames = os.listdir(base_path)
    # filenames = list(filter(lambda filename: 'X' not in filename, filenames))

    random.shuffle(filenames)

    if mode == "train":
        filenames = filenames
    else:
        filenames = filenames[:2000]

    answer_dict = {
        'X': 0,
        'A': 1,
        'B': 2,
        'C': 3,
    }
    labels = np.asarray([answer_dict[filename[-5]] for filename in filenames], dtype=np.int32)
    images = []
    for filename in filenames:
        image = cv2.imread(os.path.join(base_path, filename), 0)
        images.append(image)
    images = np.asarray(images).reshape((-1, 66 * 18)).astype(np.float32)

    return images, labels


def main(unused_argv):
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/box_convert_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50,
    )

    # train
    train_data, train_labels = data_input_fn('train')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=500,
        num_epochs=None,
        shuffle=True,
    )
    classifier.train(
        input_fn=train_input_fn,
        steps=100,
        hooks=[logging_hook]
    )

    # eval
    eval_data, eval_labels = data_input_fn('eval')
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False,
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

    def serving_input_receiver_fn():
        inputs = {"x": tf.placeholder(shape=(18, 66), dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    classifier.export_savedmodel(r"./models", serving_input_receiver_fn)


if __name__ == "__main__":
    tf.app.run()
