import vtk
import numpy as np
import math
import os
import dill
import random


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


# ver 1.2.1 추가 함수
def make_trans_offset(num_of_data, num_of_trans, trans_x, trans_y, trans_z):
    xyz_offset = [
        [
            [round(random.uniform(trans_x[0], trans_x[1]), 5),
             round(random.uniform(trans_y[0], trans_y[1]), 5),
             round(random.uniform(trans_z[0], trans_z[1]), 5)] for _ in range(num_of_trans)
        ] for i in range(num_of_data)
    ]
    # xyz_offset = np.expand_dims(np.array(xyz_offset), axis=2)
    return xyz_offset


def make_rot_matrices(num_of_data, num_of_rot, rot_x, rot_y, rot_z):
    """
    point cloud에 회전을 적용하기 위한 matrix를 계산하는 함수.
    :param num_of_data: 회전을 적용할 데이터의 갯수
    :param num_of_rot: 회전을 통해 늘릴 데이터 배수
    :param rot_x: x 축으로 회전을 적용할 범위
    :param rot_y: y 축으로 회전을 적용할 범위
    :param rot_z: z 축으로 회전을 적용할 범위
    :return: (num_of_data, num_of_rot, 3, 3)
    """
    all_rot_matrices = []
    for j in range(num_of_data):
        xyz_rot_matrix = []
        for i in range(num_of_rot):
            x_angle = random.uniform(rot_x[0], rot_x[1]) * (math.pi / 180)
            y_angle = random.uniform(rot_y[0], rot_y[1]) * (math.pi / 180)
            z_angle = random.uniform(rot_z[0], rot_z[1]) * (math.pi / 180)
            x_matrix = np.array([
                [1., 0., 0.],
                [0., math.cos(x_angle), -math.sin(x_angle)],
                [0., math.sin(x_angle), math.cos(x_angle)]])
            y_matrix = np.array([
                [math.cos(y_angle), 0., math.sin(y_angle)],
                [0., 1., 0.],
                [-math.sin(y_angle), 0., math.cos(y_angle)]])
            z_matrix = np.array([
                [math.cos(z_angle), -math.sin(z_angle), 0.],
                [math.sin(z_angle), math.cos(z_angle), 0.],
                [0., 0., 1.]])
            xy_matrix = np.dot(x_matrix, y_matrix)
            xyz_matrix = np.dot(xy_matrix, z_matrix)
            xyz_rot_matrix.append(xyz_matrix)
        all_rot_matrices.append(xyz_rot_matrix)
    return all_rot_matrices


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


def data_list_title(path):
    """
    class 객체를 불러와서 그 정보를 표시하는 함수
    :param path: './data'
    :return: 
    """
    data_list = os.listdir(path)
    for idx, data in enumerate(data_list):
        array_size = np.load(f'{path}/{data}/input_data.npy').shape
        with open(f'{path}/{data}/data_information.pkl', 'rb') as file:
            _, bone1_data, _, _ = dill.load(file)

        print(f'{data}: {array_size}, [{bone1_data.rand_sam_status}, {bone1_data.trans_status}, {bone1_data.rot_status}]')



