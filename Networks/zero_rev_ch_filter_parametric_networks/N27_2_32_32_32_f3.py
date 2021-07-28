import tensorflow as tf
import numpy as np
import tf_util


def get_model(input_tensor, is_training, bn_decay=None):
    weight_decay = 0.0
    num_point = input_tensor.get_shape()[1].value # get_shape : 텐서의 크기를 확인

    first_CNN_channels = [32, 32, 32]
    second_CNN_channels = [32, 32, 32]
    third_CNN_channels = [32, 32, 32]
    fourth_CNN_channels = [32, 32, 32]
    multi = 2


    input_tensor_transformed = tf.expand_dims(input_tensor, axis=-2)


    out1_1 = tf_util.conv2d(input_tensor_transformed, first_CNN_channels[0], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out1_2 = tf_util.conv2d(out1_1, first_CNN_channels[1], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out1_3 = tf_util.conv2d(out1_2, first_CNN_channels[2], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out1_3, axis=-2, keepdims=True)

    first_zero_tensor = tf.zeros([1, num_point, 1, first_CNN_channels[2]*multi])
    zero_net_1 = tf.concat(axis=-1, values=[net_1, first_zero_tensor])


    out2_1 = tf_util.conv2d(zero_net_1, second_CNN_channels[0], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2_2 = tf_util.conv2d(out2_1, second_CNN_channels[1], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out2_3 = tf_util.conv2d(out2_2, second_CNN_channels[2], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_2 = tf.reduce_max(out2_3, axis=-2, keepdims=True)

    second_zero_tensor = tf.zeros([1, num_point, 1, second_CNN_channels[2]*multi])
    zero_net_2 = tf.concat(axis=-1, values=[net_2, second_zero_tensor])


    out3_1 = tf_util.conv2d(zero_net_2, third_CNN_channels[0], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='three/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out3_2 = tf_util.conv2d(out3_1, third_CNN_channels[1], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='three/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out3_3 = tf_util.conv2d(out3_2, third_CNN_channels[2], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='three/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_3 = tf.reduce_max(out3_3, axis=-2, keepdims=True)

    third_zero_tensor = tf.zeros([1, num_point, 1, third_CNN_channels[2]*multi])
    zero_net_3 = tf.concat(axis=-1, values=[net_3, third_zero_tensor])


    out4_1 = tf_util.conv2d(zero_net_3, fourth_CNN_channels[0], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='four/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out4_2 = tf_util.conv2d(out4_1, fourth_CNN_channels[1], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='four/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out4_3 = tf_util.conv2d(out4_2, third_CNN_channels[2], [3, 3],
                            padding='SAME', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='four/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_4 = tf.reduce_max(out4_3, axis=-2, keepdims=True)


    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3, net_4], axis=-1), 1024, [3, 3],
                          padding='SAME', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 512, [3, 3], padding='SAME', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    # net = tf_util.conv2d(net, 256, [1,1], padding='SAME', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    # net = tf_util.conv2d(net, 128, [1,1], padding='SAME', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv3', is_dist=True)
    # net = tf_util.conv2d(net, 64, [1,1], padding='SAME', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv4', is_dist=True)
    # net = tf_util.conv2d(net, 32, [1,1], padding='SAME', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv5', is_dist=True)

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    net = tf_util.conv2d(net, 1, [3, 3], padding='SAME', stride=[1, 1],
                         activation_fn=None, scope='seg/output', is_dist=True)

    net = net[:, :, 0, 0]
    net = net

    net = tf.div(
        tf.subtract(
            net,
            tf.reduce_min(net)
        ),
        tf.subtract(
            tf.reduce_max(net),
            tf.reduce_min(net)
        )
    )

    return net



