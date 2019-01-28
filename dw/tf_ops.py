import tensorflow as tf


# ===================== convolution stuff

def conv(id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0):
    # regular conv with my favorite settings :)

    assert padding in ["SAME", "VALID", "REFLECT"], 'valid paddings are "SAME", "VALID", "REFLECT"'
    if type(size) == int:
        size = [size, size]
    if init_stddev <= 0.0:
        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    else:
        init = tf.truncated_normal_initializer(stddev=init_stddev)

    if padding == "REFLECT":
        assert size[0] % 2 == 1 and size[1] % 2 == 1, "REFLECTION PAD ONLY WORKING FOR ODD FILTER SIZE.. " + str(size)
        pad_x = size[0] // 2
        pad_y = size[1] // 2
        input = tf.pad(input, [[0, 0], [pad_x, pad_x], [pad_y, pad_y], [0, 0]], "REFLECT")
        padding = "VALID"

    return tf.layers.conv2d(input, channels, kernel_size=size, strides=[stride, stride],
                            padding=padding, kernel_initializer=init, name='conv' + id, use_bias=use_bias)


def t_conv(id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0):
    # good old t-conv. I love it!

    assert padding in ["SAME", "VALID"], 'valid paddings are "SAME", "VALID"'
    if type(size) == int:
        size = [size, size]
    if init_stddev <= 0.0:
        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    else:
        init = tf.truncated_normal_initializer(stddev=init_stddev)
    return tf.layers.conv2d_transpose(input, channels, kernel_size=size, strides=[stride, stride],
                                      padding=padding, kernel_initializer=init, name='tr_conv' + id, use_bias=use_bias)


def aconv(id, input, channels, size=3, rate=2, use_bias=True, padding="SAME", init_stddev=-1.0):
    # atrous conv. e.g. size=3 rate=2 means 5x5 filter

    assert padding in ["SAME", "VALID", "REFLECT"], 'valid paddings are "SAME", "VALID", "REFLECT"'
    if rate == 1:
        return conv(id, input, channels, size, 1, use_bias, padding, init_stddev)

    in_ch = input.get_shape().as_list()[-1]
    if init_stddev <= 0.0:
        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    else:
        init = tf.truncated_normal_initializer(stddev=init_stddev)
    filters = tf.get_variable(id + '-weights', shape=[size, size, in_ch, channels], dtype=tf.float32, initializer=init)

    if padding == "REFLECT":
        assert size[0] % 2 == 1 and size[1] % 2 == 1, "REFLECTION PAD ONLY WORKING FOR ODD FILTER SIZE.. " + str(size)
        pad_size = (size + (rate - 1) * (size - 1)) // 2
        input = tf.pad(input, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")  # NEW
        padding = "VALID"

    y = tf.nn.atrous_conv2d(input, filters, rate, padding=padding, name='aconv' + id)

    if use_bias:
        b = tf.get_variable(id + 'bias', shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        y = y + b

    return y


def t_aconv(id, input, channels, size, rate, outshape, use_bias=True, padding="SAME", init_stddev=-1.0):
    assert padding in ["SAME", "VALID", "REFLECT"], 'valid paddings are "SAME", "VALID", "REFLECT"'
    if rate == 1:
        return t_conv(id, input, channels, size, 1, use_bias, padding)  # NEW

    in_ch = input.get_shape().as_list()[-1]
    if init_stddev <= 0.0:
        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    else:
        init = tf.truncated_normal_initializer(stddev=init_stddev)
    filters = tf.get_variable(id + '-weights', shape=[size, size, in_ch, channels], dtype=tf.float32, initializer=init)

    if padding == "REFLECT":
        pad_size = (size + (rate - 1) * (size - 1)) // 2
        input = tf.pad(input, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")  # NEW
        padding = "VALID"

    y = tf.nn.atrous_conv2d_transpose(input, filters, outshape, rate, padding=padding, name='taconv' + id)
    if use_bias:
        b = tf.get_variable(id + 'bias', shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        y = y + b
    return y


def instance_norm(input):
    return tf.contrib.layers.instance_norm(input)


def dropout(id, input, is_train, keeprate=0.5):
    return tf.layers.dropout(input, rate=keeprate, training=is_train, name='dropout' + id)


def residual_unit(id, input, padding="SAME"):
    assert padding in ["SAME", "VALID"]
    channels = input.get_shape().as_list()[-1]
    if padding == "SAME":
        a1 = tf.nn.relu(conv(id + 'c1', input, channels, 3))
        a2 = tf.nn.relu(conv(id + 'c2', a1, channels, 3))
    elif padding == "VALID":
        a1 = tf.nn.relu(conv(id + 'c1', input, channels, 3, padding="VALID"))
        a2 = tf.nn.relu(transposed_conv(id + 'c2', a1, channels, 3, padding="VALID"))

    return input + a2


def resnet_block(id, input, channels, bottleneck, norm):
    if norm:
        input = instance_norm(input)
    in_channels = input.get_shape().as_list()[-1]

    c1 = conv(id + 'c1', input, bottleneck, 1)
    a2 = tf.nn.leaky_relu(c1, 0.01)
    c2 = conv(id + 'c2', a2, channels, 5)
    a3 = tf.nn.leaky_relu(c2, 0.01)
    c3 = conv(id + 'c3', a3, in_channels, 1)

    # S = tf.add_n(c3, name='add_' + id)

    return tf.nn.leaky_relu(input + c3, 0.01)


def maxpool(input, fac):
    return tf.layers.max_pooling2d(inputs=input, pool_size=[fac, fac], strides=fac)


def unpool(input, fac):
    shape = input.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(input, (shape[1] * fac, shape[2] * fac))


def batch_norm(id, is_train, input):
    return tf.layers.batch_normalization(input, training=is_train, name='bn' + id)
