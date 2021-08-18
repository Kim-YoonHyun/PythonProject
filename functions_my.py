import vtk
import numpy as np
import math

def read_stl(path, stl_name):
    """
    function of read stl file vertices and number of point information

    path: stl file path
    stl_name: stl file name
    return: stl vertices list, number of stl coordinate points
    """
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(f'{path}/{stl_name}.stl')
    file_data.Update()
    num_of_stl_points = file_data.GetOutput().GetNumberOfPoints()
    stl_vertices_list = list(
        list(file_data.GetOutput().GetPoint(i)) for i in range(num_of_stl_points)
    )
    return stl_vertices_list, num_of_stl_points

def euclidean_distance(start, end):
    """
    function of calculate the euclidean distance from start to end
    :param start: start vertex
    :param end: end vertex
    :return: euclidean distance
    """

    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
    return value


def abs_vector(vertex):
    value = np.sqrt(np.sum(np.square(vertex), axis=-1))
    return value


def calculate_point_to_line_length(center_array, start_array, target_array):
    """
    3차원 공간 상에서 center에서 target array 로 이어지는 line과 start array 간의 거리를 구하는 함수
    :param center_array:
    :param start_array:
    :param target_array:
    :return:
    """
    tar_to_start_array_vector = np.subtract(start_array, center_array)
    tar_to_target_array_vector = np.subtract(target_array, center_array)

    L = []
    for i in range(target_array.shape[0]):
        dot_product = np.dot(tar_to_start_array_vector, tar_to_target_array_vector[i])
        abs_dot_product = np.absolute(dot_product)

        cross_temp1 = abs_vector(tar_to_start_array_vector)
        cross_temp2 = abs_vector(tar_to_target_array_vector[i])
        abs_cross_product = cross_temp1 * cross_temp2
        vec_between_vec_radian = np.arccos(abs_dot_product / abs_cross_product)
        vec_between_vec_degree = vec_between_vec_radian * (180 / math.pi)

        len_of_start_array_and_center = euclidean_distance(center_array, start_array)
        L_set = np.multiply(len_of_start_array_and_center, np.sin(vec_between_vec_radian))
        L.append(min(L_set))
    L = np.array(L)
    return L