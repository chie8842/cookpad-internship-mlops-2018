"""
Food Nonfood Classifier
"""

import tensorflow as tf
import numpy as np


class FoodNonfood(object):
    def __init__(self, model_file=None):
        self.categories = ['food', 'nonfood']
        self.model_version = None
        self.load_graph()

    def load_graph(self, model_file=None):
        """load_graph

        model_file (str): model file name
        """
        if model_file is None:
            model_file = '../models/retrained_mobilenet_v2_035_224.pb'

        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        self.graph = graph
        self.model_version = model_file.split('/')[-2]

    def _get_operations(self):
        input_layer = 'Placeholder'
        output_layer = 'final_result'

        input_operation = self.graph.get_operation_by_name('import/{}'.format(input_layer))
        output_operation = self.graph.get_operation_by_name('import/{}'.format(output_layer))
        return input_operation, output_operation

    def predict(self, filename):
        """predict

        Args:
            filename (str): input image file path
        Returns:
            predict (str): "food" or "nonfood"
        """
        with tf.Session(graph = self.graph) as sess:
            t = read_tensor_from_image_file(sess, filename)
            input_operation, output_operation = self._get_operations()
            result = sess.run(
                    output_operation.outputs[0],
                    { input_operation.outputs[0]: t })
        prediction = self.categories[np.squeeze(result).argmax()]
        return prediction

def read_tensor_from_image_file(
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


