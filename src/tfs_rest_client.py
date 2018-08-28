#!/usr/bin/env python

from __future__ import print_function

import requests
import sys
import threading
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, request, redirect, jsonify, render_template, url_for

from foodnonfood import read_tensor_from_image_file
from logger import logger_config


UPLOAD_DIR = 'static/uploaded_images' # Directory to upload image files
EXTENSIONS = ['.jpeg', '.jpg', '.png'] # file extensions which can use for inference
LOGGER = logger_config('tfs_cli')


tf.app.flags.DEFINE_string('version', None, 'spacify model version')
tf.app.flags.DEFINE_boolean('debug', 'True', 'debug option')
tf.app.flags.DEFINE_string('hostport', 'localhost:8501', 'host:port of predict server')
FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR

def _predict(hostport, image_file):
    host, port = hostport.split(':')
    with tf.Session() as sess:
        image = read_tensor_from_image_file(sess, image_file)
    payload = {"signature_name": "serving_default", "instances":image.tolist()}
    url = 'http://{}:{}/v1/models/food-nonfood'.format(host, port)
    if FLAGS.version is not None:
        url = '{}/versions/{}'.format(url, FLAGS.version)
    LOGGER.info('predict url: {}'.format(url))
    r = requests.post('{}:predict'.format(url), data=json.dumps(payload))
    try:
        predictions = json.loads(r.content.decode('utf-8'))['predictions']
    except KeyError:
        LOGGER.error('prediction error')
        sys.exit(1)
    categories = ['food', 'nonfood']
    category = categories[np.squeeze(predictions).argmax()]
    return category

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """upload_file
    This function is for prototyping.
    A user can upload your picture and this application returns the result of food-nonfood classification.
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'img_file' not in request.files:
            return redirect(request.url)
        image = request.files['img_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if image.filename == '':
            return redirect(request.url)

        # save image_file
        save_path = os.path.join(app.config['UPLOAD_DIR'], image.filename)
        LOGGER.debug(save_path)
        if os.path.isfile(save_path):
            os.remove(save_path)
        image.save(save_path)

        category = _predict(FLAGS.hostport, save_path)
        return render_template('index.html', filename=save_path, category=category)
    return render_template('index.html')


def main(_):
    app.run(host='0.0.0.0', port=5000, debug=FLAGS.debug)

if __name__ == '__main__':
  tf.app.run()
