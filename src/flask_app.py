#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Machine Learning Application with flask.
It has Web Application function with simple UI and also API

Run application:
```
python flask_app.py
```

To access this application
Usage1:
    For Prototyping
    access to  `http://hostname:5000` with browser
Usage2: API
     Use predict API
 ```
$ curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/v1/api/predict
```

Use custom model
1. specify model when to start application

```
python flask_app.py --model_path /path/to/model_file
```

2. change model while application is started
```
curl -X POST http://localhost:5000/v1/api/reload_model -H "Content-Type: application/json" \
-d '{"model_path": "/path/to/model_path"}'
```
"""

import os
import sys

from flask import Flask, request, redirect, jsonify, render_template, url_for
import tensorflow as tf

from foodnonfood import FoodNonfood
from logger import logger_config

UPLOAD_DIR = 'static/uploaded_images' # Directory to upload image files
EXTENSIONS = ['.jpeg', '.jpg', '.png'] # file extensions which can use for inference
FOODNONFOOD = FoodNonfood() # Initialization of FoodNonfood classifier
LOGGER = logger_config('flask_app')

tf.app.flags.DEFINE_boolean('debug', 'True', 'debug option')
tf.app.flags.DEFINE_string('model_path', None, 'file path of custom model')
FLAGS = tf.app.flags.FLAGS


app = Flask(__name__)

app.config['UPLOAD_DIR'] = UPLOAD_DIR

def _make_upload_dir():
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

def _check_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext in EXTENSIONS

def _predict(image):
    if image and _check_extension(image.filename):
        LOGGER.debug(image.filename)
        save_path = os.path.join(app.config['UPLOAD_DIR'], image.filename)
        LOGGER.debug(save_path)
        if os.path.isfile(save_path):
            os.remove(save_path)
        image.save(save_path)
        category = FOODNONFOOD.predict(save_path)
        return category, save_path
    else:
        return '', None

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
        category, save_path = _predict(image)
        LOGGER.info(save_path)
        return render_template('index.html', filename=save_path, category=category)
    return render_template('index.html')

@app.route('/v1/api/predict', methods=['POST'])
def predict():
    """predict
    This function provides predict API of food/nonfood
    """
    if request.method == 'POST':
        img_file = request.files['image']
        category, _ = _predict(img_file)
        LOGGER.debug('category of {}: {}'.format(img_file, category))
        message = {'category': category}
        return jsonify(message)

@app.route('/v1/api/reload_model', methods=['POST', 'GET'])
def reload_model():
    """reload_model
    Users can reload foodnonfood model file with this function.
    """
    if request.method == 'POST':
        model_file = request.form['model_path']
        model.load_graph(model_file)
        LOGGER.debug('current model is {}'.format(model_file))

def main(_):
    # create directory for uploaded image if not existed
    _make_upload_dir()
    LOGGER.debug('created {}'.format(UPLOAD_DIR))

    # load graph if model file is existed
    if FLAGS.model_path:
        FOODNONFOOD.load_graph(FLAGS.model_path)

    # start application server
    app.run(host='0.0.0.0', port=5000, debug=FLAGS.debug)


if __name__ == '__main__':
    tf.app.run()
