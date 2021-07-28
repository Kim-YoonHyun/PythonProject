import tensorflow as tf
import numpy as np
import tf_util




def get_model(input_tensor, is_training, bn_decay=None):
    weight_decay = 0.0
    num_point = input_tensor.get_shape()[1].value # get_shape : 텐서의 크기를 확인

    k = 30

    input_tensor_transformed = input_tensor

    # Transform Net
    adj_matrix = tf_util.pairwise_distance(input_tensor_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(input_tensor_transformed, nn_idx=nn_idx, k=k)

    out1_1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out1_2 = tf_util.conv2d(out1_1, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out1_3 = tf_util.conv2d(out1_2, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='one/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out1_3, axis=-2, keepdims=True)

    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

    out2_1 = tf_util.conv2d(edge_feature, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2_2 = tf_util.conv2d(out2_1, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv2', bn_decay=bn_decay, is_dist=True)

    net_2 = tf.reduce_max(out2_2, axis=-2, keepdims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2])

    # CONV
    net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv3', is_dist=True)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv4', is_dist=True)
    net = tf_util.conv2d(net, 32, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv5', is_dist=True)

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    net = tf_util.conv2d(net, 1, [1, 1], padding='VALID', stride=[1, 1],
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



