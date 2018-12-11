import tensorflow as tf

def conv_layer(x, filter_shape, stride):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding='SAME')

def conv_layer_depth(x, filter_shape1, filter_shape2, stride):
    filters1 = tf.get_variable(
        name='weight1',
        shape=filter_shape1,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2 = tf.get_variable(
        name='weight2',
        shape=filter_shape2,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.separable_conv2d(x, filters1, filters2, [1, stride, stride, 1], padding='SAME')


def dilated_conv_layer(x, filter_shape, dilation):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')


def dilated_sep_conv_layer(x, filter_shape1, filter_shape2, dilation):
    filters1 = tf.get_variable(
        name='weight1',
        shape=filter_shape1,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2 = tf.get_variable(
        name='weight2',
        shape=filter_shape2,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.separable_conv2d(x, filters1, filters2, [1, 1, 1, 1], padding='SAME', rate=[dilation, dilation])


def deconv_layer(x, filter_shape, output_shape, stride):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1])

def deconv_layer_BL_sep(x, filter_shape1, filter_shape2, stride):
    filters1 = tf.get_variable(
        name='weight1',
        shape=filter_shape1,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2 = tf.get_variable(
        name='weight2',
        shape=filter_shape2,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    x = tf.image.resize_bilinear(x, [x.shape[1]*stride, x.shape[2]*stride])
    return tf.nn.separable_conv2d(x, filters1, filters2, [1, 1, 1, 1], padding='SAME')

def deconv_layer_PS_sep(x, filter_shape1, filter_shape2, stride):
    def _phase_shiftr(I, r):
       # Helper function with main phase shift operation
       bsize, a, b, c = I.get_shape().as_list()
       X = tf.reshape(I, (I.shape[0], a, b, r, r))
       X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
       X = tf.concat([tf.squeeze(x,1) for x in X], 2)  # bsize, b, a*r, r
       X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
       X = tf.concat([tf.squeeze(x,1) for x in X], 2)  # bsize, a*r, b*r
       return tf.reshape(X, (I.shape[0], a*r, b*r, 1))

    def PSr(X, r):
       # Main OP that you can arbitrarily use in you tensorflow code
       Xc = tf.split(X, int(int(X.shape[3])/4), 3)
       X = tf.concat([_phase_shiftr(x, r) for x in Xc], 3 ) # Do the concat RGB
       return X
    
    filters1 = tf.get_variable(
        name='weight1',
        shape=filter_shape1,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2 = tf.get_variable(
        name='weight2',
        shape=filter_shape2,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    x = tf.nn.separable_conv2d(x, filters1, filters2, [1, 1, 1, 1], padding='SAME')
    return PSr(x, stride)

def deconv_layer_PS(x, filter_shape, stride):
    def _phase_shiftr(I, r):
       # Helper function with main phase shift operation
       bsize, a, b, c = I.get_shape().as_list()
       X = tf.reshape(I, (I.shape[0], a, b, r, r))
       X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
       X = tf.concat([tf.squeeze(x,1) for x in X], 2)  # bsize, b, a*r, r
       X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
       X = tf.concat([tf.squeeze(x,1) for x in X], 2)  # bsize, a*r, b*r
       return tf.reshape(X, (I.shape[0], a*r, b*r, 1))

    def PSr(X, r):
       # Main OP that you can arbitrarily use in you tensorflow code
       Xc = tf.split(X, int(int(X.shape[3])/4), 3)
       X = tf.concat([_phase_shiftr(x, r) for x in Xc], 3 ) # Do the concat RGB
       return X
    
    filters = tf.get_variable(
        name='weight1',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], padding='SAME')
    return PSr(x, stride)



def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

