import numpy as np
import pandas as pd
import math

print(162*500*162*900)


print((10**6)*10000)
exit()
# a = np.array([
#     [[0, 0, 0, 0, 0]],
#     [[1, 1, 1, 1, 1]]
# ])
# print(a)
# b = np.arange(0, 70).reshape(2, 7, 5)
# print(b)
# c = np.multiply(a, b)
# print(c)
#
# #
# exit()




def abs_vector(vertex):
    value = np.sqrt(np.sum(np.square(vertex), axis=-1))
    return value


def euclidean_distance(start, end):
    """
    function of calculate the euclidean distance from start to end
    :param start: start vertex
    :param end: end vertex
    :return: euclidean distance
    """
    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
    return value


np.random.seed(2021)
start_array = np.random.randint(-20, 20, 63).reshape(3, 7, 3)
target_array = np.random.randint(-3, 3, 9).reshape(3, 1, 3)
side_array = np.random.randint(-15, 15, 45).reshape(3, 5, 3)

def calculate_point_to_line_length2(start_array, target_array, side_array):
    """
    center >>> start
    start >>> side
    3차원 공간 상에서 center에서 target array 로 이어지는 line과 start array 간의 거리를 구하는 함수
    :param center_array:
    :param start_array:
    :param target_array:
    :return:
    """
    side_to_target_array_vector = np.subtract(side_array, target_array)
    start_to_target_array_vector = np.subtract(start_array, target_array)
    # print(side_to_target_array_vector)
    # print(start_to_target_array_vector)
    # print()
    # print(side_to_target_array_vector.shape)
    # print(start_to_target_array_vector.T.shape)
    dot_product = np.dot(side_to_target_array_vector, start_to_target_array_vector.T)
    # print(dot_product)
    dot_product_trans = dot_product.T
    abs_dot_product_trans = np.absolute(dot_product_trans)
    # print(abs_dot_product_trans)
    cross_temp1 = abs_vector(side_to_target_array_vector)
    # print(cross_temp1)
    cross_temp2 = np.expand_dims(abs_vector(start_to_target_array_vector), axis=-1)
    # print(cross_temp2)
    abs_cross_product = np.multiply(cross_temp1, cross_temp2)
    # print(abs_cross_product)
    vec_between_vec_radian = np.arccos(abs_dot_product_trans / abs_cross_product)
    vec_between_vec_degree = vec_between_vec_radian * (180 / math.pi)

    # print(vec_between_vec_degree)
    len_of_target_array_and_side = euclidean_distance(target_array, side_array)
    # print(len_of_target_array_and_side)
    # print(np.sin(vec_between_vec_radian))
    l_set = np.multiply(len_of_target_array_and_side, np.sin(vec_between_vec_radian))
    # print(l_set)
    min_l_set = np.minimum.reduce(l_set, axis=-1)
    return min_l_set

l = []
for i in range(start_array.shape[0]):
    min_l = calculate_point_to_line_length2(start_array[i], target_array[i], side_array[i]).tolist()
    l.append(min_l)
print(np.array(l))

def calculate_point_to_line_length3(start_array, target_array, side_array):
    side_to_target_array_vector = np.subtract(side_array, target_array)

    start_to_target_array_vector = np.subtract(start_array, target_array)
    start_to_target_array_vector_trans = np.swapaxes(start_to_target_array_vector, 1, 2)

    dot_product = np.dot(side_to_target_array_vector, start_to_target_array_vector_trans)
    ai = np.arange(0, start_array.shape[0]).reshape(start_array.shape[0], 1, 1, 1)
    dot_product_take = np.squeeze(np.take_along_axis(dot_product, ai, axis=2))
    dot_product_take_trans = np.swapaxes(dot_product_take, 1, 2)
    abs_dot_product_trans = np.absolute(dot_product_take_trans)

    side_to_target_array_vector_abs = abs_vector(side_to_target_array_vector)
    start_to_target_array_vector_abs = np.expand_dims(abs_vector(start_to_target_array_vector), axis=-1)
    start_to_target_array_vector_abs = np.expand_dims(np.concatenate(start_to_target_array_vector_abs, axis=-2), axis=-1)

    abs_cross_product = np.multiply(side_to_target_array_vector_abs, start_to_target_array_vector_abs)
    abs_cross_product = abs_cross_product.reshape(start_array.shape[0], start_array.shape[1], start_array.shape[0], side_array.shape[1])
    abs_cross_product = np.squeeze(np.take_along_axis(abs_cross_product, ai, axis=2))

    vec_between_vec_radian = np.arccos(abs_dot_product_trans / abs_cross_product)
    vec_between_vec_degree = vec_between_vec_radian * (180 / math.pi)

    len_of_target_array_and_side = np.expand_dims(euclidean_distance(target_array, side_array), axis=-2)
    l_set = np.multiply(len_of_target_array_and_side, np.sin(vec_between_vec_radian))
    min_l_set = np.minimum.reduce(l_set, axis=-1)
    return min_l_set
min_l_set = calculate_point_to_line_length3(start_array, target_array, side_array)
print(min_l_set)