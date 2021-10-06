# edit this file to create a simple model for your dataset.

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(3, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    return self.dense3(x)

iris_model = MyModel(name='iris_model')
