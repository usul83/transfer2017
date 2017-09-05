# Graeme Blyth 2017

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import numpy as np
import pandas as pd


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    initial = tf.constant(0.1 , shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool22(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def max_pool44(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')


def flatten_outer_dims(logits):
  """Flattens logits' outer dimensions and keep its last dimension."""
  # from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/nn_ops.py
  rank = array_ops.rank(logits)
  last_dim_size = array_ops.slice(array_ops.shape(logits), [math_ops.subtract(rank, 1)], [1])
  output = array_ops.reshape(logits, array_ops.concat([[-1], last_dim_size], 0))

  # Set output shape if known.
  shape = logits.get_shape()
  if shape is not None and shape.dims is not None:
    shape = shape.as_list()
    product = 1
    product_valid = True
    for d in shape[:-1]:
      if d is None:
        product_valid = False
        break
      else:
        product *= d
    if product_valid:
      output_shape = [product, shape[-1]]
      output.set_shape(output_shape)
  return output


def compute_fisher(y,varList,batchSize,zeta):
    # compute Fisher information for a list of variables
    Fout = []
    dim = []
    for var in range(len(varList)):
        dim.append(varList[var].get_shape().as_list()[-1])
        Fout.append(tf.zeros([dim[var],dim[var]]))
    for imLoop in range(batchSize):
        dydw = tf.gradients(tf.log(y[imLoop,:] + zeta), varList)
        for var in range(len(Fout)):
            dydw[var] = flatten_outer_dims(dydw[var])
            Fout[var] = tf.add(Fout[var],tf.matmul(dydw[var],dydw[var],True) / batchSize)
    for var in range(len(Fout)):
        Fout[var] = Fout[var] + np.identity(dim[var]) * zeta
        Fout[var] = tf.divide(Fout[var], tf.trace(Fout[var]))
    return Fout


def compute_fisher2(y,varList,batchSize,currTasks,zeta):
    # this version samples a random class for each gradient calc and is quicker if there are only a few relevant classes
    Fout = []
    dim = []
    for var in range(len(varList)):
        dim.append(varList[var].get_shape().as_list()[-1])
        Fout.append(tf.zeros([dim[var], dim[var]]))
    for imLoop in range(batchSize):
        randClass = tf.random_uniform([], 0, tf.shape(currTasks)[0], dtype=tf.int32)
        randClass = tf.cast(currTasks[randClass], dtype=tf.int32)
        dydw = tf.gradients(tf.log(y[imLoop, randClass]), varList)
        for var in range(len(Fout)):
            dydw[var] = flatten_outer_dims(dydw[var])
            Fout[var] = tf.add(Fout[var], tf.matmul(dydw[var], dydw[var], True) / batchSize)
    for var in range(len(Fout)):
        Fout[var] = Fout[var] + np.identity(dim[var]) * zeta
        Fout[var] = tf.divide(Fout[var], tf.trace(Fout[var]))
    return Fout


def softmax(x):
    # softmax for vectors in N rows, returns matrix of same shape where rows add to 1
    denom = np.sum(np.exp(x), axis=1)
    return np.exp(x) / denom[:, None]


def TaskBatch(full_xs, full_ys, task):
    # return batches containing only three different MNIST digits
    batch_xs = full_xs[full_ys[:, task[0]] == 1]
    batch_ys = full_ys[full_ys[:, task[0]] == 1]
    batch_xs = np.append(batch_xs, full_xs[full_ys[:, task[1]] == 1], axis=0)
    batch_ys = np.append(batch_ys, full_ys[full_ys[:, task[1]] == 1], axis=0)
    batch_xs = np.append(batch_xs, full_xs[full_ys[:, task[2]] == 1], axis=0)
    batch_ys = np.append(batch_ys, full_ys[full_ys[:, task[2]] == 1], axis=0)
    return batch_xs, batch_ys

def TaskBatch2(df, imageSize, task):
    # return batches containing only three different MNIST2 digits
    full = df.as_matrix()
    batch = full[full[:, imageSize + task[0]] == 1]
    batch = np.append(batch, full[full[:, imageSize + task[1]] == 1], axis=0)
    batch = np.append(batch, full[full[:, imageSize + task[2]] == 1], axis=0)
    return pd.DataFrame(batch)

