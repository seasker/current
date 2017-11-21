import tensorflow as tf 
import numpy as np
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
  tf.constant
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
  if tf.tf.shape(input)[0]output_shape[0]:
    print('ssssss')
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




ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
indices = tf.constant([[4], [3], [3] ,[7]])
updates = tf.constant([9, 10, 11, 12])
update = tf.scatter_nd_update(ref, indices, updates)

with tf.device('/gpu:0'):
    a=tf.Variable(tf.reshape(tf.range(120,dtype=tf.float32),[2,3,4,5]))
    b,arg=tf.nn.max_pool_with_argmax(a,[1,2,2,1],[1,2,2,1],padding='SAME')
    sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
    c=max_unpool2d(b,[2,4,5,5],[2,2],[2,2],padding='Valid')
    sess.run(tf.global_variables_initializer())
    #=sess.run((b,arg))
    u=sess.run(update)
    print(a.get_shape())
    b_,t,c_=sess.run((b,arg,c))
print(b_)
print(b_.shape)
print('')
print(t)
print(u)



print(c_)
print(c_.shape)
print(c.shape)