# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python

"""A client that talks to tensorflow_model_server loaded with foodnonfood model.
The client queries the service with a test image to get predictions,
and calculates the inference error rate.
Typical usage example:
    python tfs_grpc_client.py --server=localhost:8501 --image_path=/path/to/image.jpg
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

RPC_TIMEOUT = 5.0

tf.app.flags.DEFINE_string('hostport', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('image_path', 'data/test_images/food.jpg', 'Image file path')
FLAGS = tf.app.flags.FLAGS

def _read_tensor_from_image_file(
        sess,
        file_name,
        input_height=224,
        input_width=224,
        channels=3,
        input_mean=0,
        input_std=255):
    file_reader = tf.read_file(file_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels=channels)
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
            dims_expander,
            [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    result = sess.run(normalized)
    return result

def predict(hostport):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
  Returns:
    category: "food" or "nonfood"
    The classification error rate.
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  image_file = FLAGS.image_path
  with tf.Session() as sess:
      image = _read_tensor_from_image_file(sess, image_file)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'food-nonfood'
  request.model_spec.signature_name = 'serving_default'

  request.inputs['image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image, dtype=dtypes.float32, shape=[1, 224, 224, 3]))
  response = stub.Predict(request, RPC_TIMEOUT)  # 5 seconds
  result = np.array(response.outputs['prediction'].float_val)
  categories = ['food', 'nonfood']
  category = categories[np.squeeze(result).argmax()]
  return category

def main(_):
  result_future = predict(FLAGS.hostport)
  print('\nInference: {}'.format(result_future))


if __name__ == '__main__':
  tf.app.run()

