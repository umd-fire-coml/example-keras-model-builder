# edit this file to create a simple model for your dataset.

import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.r1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.r2 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
    self.sm1 = tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    self.sm2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = inputs
    for _ in range(5):
      x = self.r1(inputs)
      x = self.sm1(inputs)
      x = self.r2(inputs)
      x = self.sm2(inputs)
    return x
