from BasicConvLSTMCell import BasicConvLSTMCell
import tensorflow as tf 
sess=tf.Session()
batch_size=4
max_step=1000
image_size=[35,288]
initial_state_size=image_size
x=tf.placeholder(tf.float32,shape=[batch_size]+image_size+[3])
history_N=7

def variable_with_decay(name, shape,initializer, 
                                decay=None,
                                loss_name='losses'):
    var = tf.get_variable(name=name,
                             shape=shape,
                             initializer=initializer)
    if decay is not None:
        var_decay = tf.multiply(
            tf.nn.l2_loss(var), decay, name='var_loss')
        tf.add_to_collection(name=loss_name, value=var_decay)
    return var


def variable_with_regularizer(name, shape, initializer, 
                                        regularizer=None,
                                        loss_name='losses'):
    var = tf.get_variable(name=name,
                             shape=shape,
                             initializer=initializer)
    if regularizer is not None:
        tf.add_to_collection(name=loss_name, value=regularizer(var))
    return var

def print_activation_info(activation):
    print(activation.op.name, ' ', activation.get_shape().as_list())

def conv2d(input,num_output,ksize,
            strides=[1,1],
            padding='SAME',
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            activaion_fn=tf.nn.relu,
            scope=None,
            resuse=False,
            params=None):
    num_input=input.get_shape()[-1].value
    with tf.variable_scope(scope or ' Conv'):
        kernel=variable_with_decay('kernel',shape=[ksize[0],ksize[1],num_input,num_output],initializer=initializer)
        bias=tf.get_variable('bias',shape=[num_output],initializer=tf.constant_initializer(0.0))
        conv=tf.nn.conv2d(input,kernel,strides=[1,strides[0],strides[0],1],padding=padding)
        conv_op=activation_fn(tf.nn.bias_add(conv,bias))
        if params is not None:
            params+=[kernel,bias]
        return conv_op,params    


def max_pool2d(input,ksize,strides,padding='SAME'):
    mpool_op=tf.nn.max_pool(input,[1,ksize[0],ksize[1],1],strides=[1,strides[0],stride[1],1],padding=padding)
    return mpool_op

def deconv2d(input,output_shape,ksize,
            strides=[1,1],
            padding='SAME',
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            activaion_fn=tf.nn.relu,
            scope=None,
            resuse=False,
            params=None):
    # filter : [height, width, output_channels, in_channels]
    kernel = variable_with_decay('kernel', 
                              [ksize[0], ksize[1], output_shape[-1],input.get_shape()[-1]],
                              initializer=initalizer)
    
    try:
      deconv = tf.nn.conv2d_transpose(input, kernel,
                                      output_shape=output_shape,
                                      strides=[1,strides[0], strides[1], 1],
                                      padding=padding)
    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                          strides=[1, d_h, d_w, 1])

    bias = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    deconv_op = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if params is not None:
            params+=[kernel,bias]
    return deconv_op

def Up_pooling2d_with_zero_filled(input):
  output = tf.concat(axis=3, values=[input, tf.zeros_like(input)])
  output = tf.concat(axis=2, values=[output, tf.zeros_like(output)])

  input_shape = input.get_shape().as_list()
  if None not in sh[1:]:
    output_shape = [-1, input_shape[1] * 2, input_shape[2] * 2, input_shape[3]]
    return tf.reshape(output, output_shape)
  else:
    input_shape = tf.shape(input)
    return tf.reshape(output, [-1, input_shape[1] * 2, input_shape[2] * 2, input_shape[3]])




def forward(input,scope=None):
    with tf.variable_scope(scope or 'SimpleLSTMNet'):
        state=tf.zeros([batch_size]+initial_state_size+[64])
        cell=BasicConvLSTMCell(initial_state_size, [3, 3],32)
        for n in range(history_N):
            lstm_h,new_state=cell(input,state)
        conv1_1=conv2d(lstm_h,64,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_1')
        conv1_2=conv2d(conv1_1,64,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_2')
        pool1=max_pool2d(conv1_2,[2,2],strides=[2,2])
        conv2_1=conv2d(pool1,128,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_1')
        conv2_2=conv2d(conv2_1,128,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_2')
        pool2=max_pool2d(conv2_2,[2,2],strides=[2,2])
        conv3_1=conv2d(pool2,128,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_1')
        conv3_2=conv2d(conv3_1,128,[3,3],initializer=tf.truncated_normal_initializer(stddev=0.01),scope='Conv1_2')
        pool3=max_pool2d(conv3_2,[2,2],strides=[2,2])
        up



    

def loss    
`


def train():
    for i in range(max_step):
        sess.run(logits)

    








