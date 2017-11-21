import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


def batch_norm(inputs, name, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
                                      reuse=reuse,scope=name,scale=True)


def conv2d(input_, output_dim, 
            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
            name="conv2d", reuse=False, padding='SAME'):
   with tf.variable_scope(name, reuse=reuse):
     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                         initializer=tf.contrib.layers.xavier_initializer())
     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
 
     biases = tf.get_variable('biases', [output_dim],
                              initializer=tf.constant_initializer(0.0))
     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
 
     return conv


def deconv2d(input_, output_shape,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="deconv2d", reuse=False, with_w=False, padding='SAME'):
  with tf.variable_scope(name, reuse=reuse):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_h, output_shape[-1],
                              input_.get_shape()[-1]],
                        initializer=tf.contrib.layers.xavier_initializer())
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w,
                                      output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1],
                                      padding=padding)

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                          strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def relu(x):
  return tf.nn.relu(x)


def tanh(x):
  return tf.nn.tanh(x)

def sigmoid(x):
  return tf.nn.sigmoid(x) 


def shape2d(a):
  """
  a: a int or tuple/list of length 2
  """
  if type(a) == int:
      return [a, a]
  if isinstance(a, (list, tuple)):
      assert len(a) == 2
      return list(a)
  raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


def UnPooling2x2ZeroFilled(x):
  out = tf.concat(axis=3, values=[x, tf.zeros_like(x)])
  out = tf.concat(axis=2, values=[out, tf.zeros_like(out)])

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    return tf.reshape(out, out_size)
  else:
    sh = tf.shape(x)
    return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


def MaxPooling(x, shape, stride=None, padding='SAME'):
  """
  MaxPooling on images.
  :param input: NHWC tensor.
  :param shape: int or [h, w]
  :param stride: int or [h, w]. default to be shape.
  :param padding: 'valid' or 'same'. default to 'valid'
  :returns: NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)

tf.scatter_nd_add

def max_pool2d_with_argmax(x, shape, stride=None, padding='SAME'):
  """
  MaxPooling on images.
  :param input: NHWC tensor.
  :param shape: int or [h, w]
  :param stride: int or [h, w]. default to be shape.
  :param padding: 'valid' or 'same'. default to 'valid'
  :returns: NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool_argmax(x, ksize=shape, strides=stride, padding=padding)



def max_unpool2d_with_argmax(input,output_shape,argmax,kssize,sstrides,padding='SAME',scope=None):
  input_height,input_width=input.get_shape()[1:3].as_list()
  print(input_height,input_width)
  output_height,output_width=tf.constant(0,shape=output_shape).get_shape()[1:3].as_list()
  print(output_height,output_width)
  if padding=='SAME':
    assert np.ceil(output_height/sstrides[0])==input_height,'output_shape[1]：height is incompatible '
    assert np.ceil(output_width/sstrides[1])==input_width,'output_shape[2]:width is incompatible'
  else:
    assert np.ceil((output_height-kssize[0]+1)/sstrides[0])==input_height,'output_shape[1]：height is incompatible '
    assert np.ceil((output_width-kssize[1]+1)/sstrides[1])==input_width,'output_shape[2]:width is incompatible'
  with tf.variable_scope(scope or "max_unpool"):
    input_shape=tf.shape(input)
    flat_input_size=input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    flat_input=tf.reshape(input,[flat_input_size])
    batch_ind_range=tf.reshape(tf.range(output_shape[0], dtype=argmax.dtype), shape=[input_shape[0], 1, 1, 1])
    batch_ind=tf.ones_like(argmax) * batch_ind_range
    batch_ind=tf.reshape(batch_ind,[flat_input_size, 1])
    ind=tf.reshape(argmax, [flat_input_size, 1])
    ind=tf.concat([batch_ind,ind],1)
    max_unpool_op=tf.scatter_nd( ind, flat_input,[output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])
    return max_unpool_op


def max_unpool2d(input,output_shape,kssize,sstrides,padding='SAME',scope=None):
  input_height,input_width=input.get_shape()[1:3].as_list()
  print(input_height,input_width)
  output_height,output_width=tf.constant(0,shape=output_shape).get_shape()[1:3].as_list()
  print(output_height,output_width)
  if padding=='SAME':
    assert np.ceil(output_height/sstrides[0])==input_height,'output_shape[1]：height is incompatible '
    assert np.ceil(output_width/sstrides[1])==input_width,'output_shape[2]:width is incompatible'
    
  else:
    assert np.ceil((output_height-kssize[0]+1)/sstrides[0])==input_height,'output_shape[1]：height is incompatible '
    assert np.ceil((output_width-kssize[1]+1)/sstrides[1])==input_width,'output_shape[2]:width is incompatible'
  with tf.variable_scope(scope or "max_unpool"):
    input_shape=tf.shape(input)
    flat_input_size=input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    flat_input=tf.reshape(input,[flat_input_size])
    batch_ind_range=tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=[input_shape[0], 1, 1, 1])
    batch_ind=tf.ones_like(input,dtype=tf.int32) * batch_ind_range
    batch_ind=tf.reshape(batch_ind,[flat_input_size, 1])

    height_ind_range=tf.reshape(tf.range(input_shape[1],dtype=tf.int32),shape=[input_shape[1],1,1])
    height_ind=tf.ones_like(input,dtype=tf.int32)*height_ind_range * sstrides[0]
    height_ind=tf.reshape(height_ind,[flat_input_size, 1])

    width_ind_range=tf.reshape(tf.range(input_shape[2],dtype=tf.int32),shape=[input_shape[2],1])
    width_ind=tf.ones_like(input,dtype=tf.int32)*width_ind_range * sstrides[1]
    width_ind=tf.reshape(width_ind,[flat_input_size, 1])

    channel_ind_range=tf.range(input_shape[3],dtype=tf.int32)
    channel_ind=tf.ones_like(input,dtype=tf.int32)*channel_ind_range
    channel_ind=tf.reshape(channel_ind,[flat_input_size, 1])
    ind=tf.concat([batch_ind,height_ind,width_ind,channel_ind],1)
    max_unpool_op=tf.scatter_nd( ind, flat_input,[output_shape[0], output_shape[1],output_shape[2] , output_shape[3]])
    return max_unpool_op



#@layer_register()
def FixedUnPooling(x, shape):
  """
  Unpool the input with a fixed mat to perform kronecker product with.
  :param input: NHWC tensor
  :param shape: int or [h, w]
  :returns: NHWC tensor
  """
  shape = shape2d(shape)
  
  # a faster implementation for this special case
  return UnPooling2x2ZeroFilled(x)


def gdl(gen_frames, gt_frames, alpha):
  """
  Calculates the sum of GDL losses between the predicted and gt frames.
  @param gen_frames: The predicted frames at each scale.
  @param gt_frames: The ground truth frames at each scale
  @param alpha: The power to which each gradient term is raised.
  @return: The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(3), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  strides = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)

  gdl_loss = tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha))

  # condense into one tensor and avg
  return gdl_loss


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0,
           reuse=False, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

