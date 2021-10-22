import tensorflow as tf
from tensorflow.keras.datasets import mnist
import sklearn
sklearn.feature_selection.mutual_info_regression

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name ='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name ='y-input')

with tf.name_scope('input_reshape'):


(x_train, y_train),(x_test, y_test) = mnist.load_data()