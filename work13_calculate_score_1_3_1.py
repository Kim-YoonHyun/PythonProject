# _2
# 복잡한 계산 간결화

# _3
# score 최적점을 0 에서 1 로 변경
# _1
# epsilon 을 l1, l2, l3 각각의 최댓값으로 설정함

# _4
# word 설명과 맞추기
# _4_1
# 실행전후 파라미터 변경, 편의성 수정
# Functions import 방식 변경

# _5
# 코드 renewal
# pandas, class 등 추가 사용
# data_list_title 함수 사용

# 1_3_1
# call_data_version, call_dataframe, call_other_information, call_data_title, call_data_class_information 함수 추가 20210903
# calculate_point_to_line_length, abs_vector, euclidean_distance 함수를 옮겨옴 20210903
# calculate_point_to_line_length 함수의 계산 시간 단축 필요 20210903

import numpy as np
import time
import os
import dill
import pandas as pd
import math


def call_data_version(path):
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('ver'):
            return file
def call_dataframe(path):
    df = pd.read_csv(f'{path}/data_information.csv')
    df.drop(columns='Unnamed: 0', inplace=True)
    df.fillna('0', inplace=True)
    df.set_index('data_name', inplace=True)
    return df
def call_other_information(path):
    with open(f'{path}/other_information.txt', 'r') as file:
        line = file.readlines()
    other_information = dict(eval('{' + ', '.join(line).replace('\n', '') + '}'))
    return other_information
def call_data_title(path):
    data_list = os.listdir(path)
    for data_name in data_list:
        data_path = f'{path}/{data_name}'
        data_version = call_data_version(data_path)
        input_data = np.load(f'{data_path}/input_data.npy')

        other_information = call_other_information(data_path)
        set_range = other_information['stl set range']

        data_df = call_dataframe(data_path)
        rand_sam = data_df.rand_sam_status[1]
        rot = data_df.rot_status[0]
        trans = data_df.trans_status[0]

        print(f'{data_name:>7s} ({data_version}):{str(input_data.shape):>15s}, {set_range:>8s}, {rand_sam:>8s}, {rot:>7s}, {trans:>18s}')
def call_data_class_information(path):
    with open(f'{path}/data_information.pkl', 'rb') as file:
        return dill.load(file)

def print_data_information(*args):
    for data in args:
        print(data, '\n')

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


def main():
    while True:
        # <title>------------------------------------------------------------
        base_path = f'data'
        call_data_title(base_path)
        data_num = int(input('\n 계산할 데이터 번호 입력 > '))
        if data_num == 10000:
            break
        switch = int(input('\nscore type 선택. 1: Max, 2: Min > '))
        if switch == 1:
            score_type = 'Max'
        else:
            score_type = 'Min'

        # <score calculate>---------------------------------------------------------------------
        print(f'\nscore calculate...')
        start = time.time()

        data_path = f'{base_path}/data{data_num}'
        input_data = np.load(f'{data_path}/input_data.npy')
        target_data, bone1_data, bone2_data, skin_data = call_data_class_information(data_path)
        print_data_information(target_data, bone1_data, bone2_data, skin_data)

        # Bone1 및 Bone2 와 l1 간의 모든 거리 중 최소값(l2와 가장 가까운 Bone 좌표)구하기(l2, l3)
        l1 = euclidean_distance(start=skin_data.data_vertices_list, end=target_data.data_vertices_list)
        l2 = []
        l3 = []
        for idx, data in enumerate(input_data):
            print(f'{idx + 1}/{input_data.shape[0]}')
            l2.append(calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                     np.array(bone1_data.data_vertices_list[idx]),
                                                     np.array(skin_data.data_vertices_list[idx])))
            l3.append(calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                     np.array(bone2_data.data_vertices_list[idx]),
                                                     np.array(skin_data.data_vertices_list[idx])))

        l1_e = np.add(l1, np.expand_dims(np.maximum.reduce(l1, axis=1), axis=1))
        l2_e = np.add(l2, np.expand_dims(np.maximum.reduce(l2, axis=1), axis=1))
        l3_e = np.add(l2, np.expand_dims(np.maximum.reduce(l3, axis=1), axis=1))

        if score_type == 'Max':
            score = np.divide(np.maximum(l2_e, l3_e), l1_e)
        elif score_type == 'Min':
            score = np.divide(l1_e, np.maximum(l2_e, l3_e))
        else:
            print(f'there is no score type "{score_type}"')
            continue

        print('run time :', round(time.time() - start, 2), 'sec')
        np.save(f'{data_path}/score', score)
        with open(f'{data_path}/score_type_{score_type}.txt', 'w') as file:
            file.write(score_type)
        # np.save(f'{data_path}/score function type={score_type}', score_type)
        print('\n')
    exit()


if __name__ == '__main__':
    main()
#
#
#
# # ############################## make file #############################################################################
#     np.save(f'{path}/score', score)
#     np.save(f'{path}/score function type', switch)
#     file = open(f'{path}/information.txt', 'a', encoding= 'utf8')
#     file.write(f'\n\n\nscore function : {score_func}\n\n')
#     print('\n\n추가 정보 기입')
#     string = str(input())
#     file.write(f'추가 정보: {string}')
#     file.close()
#     print('\n\n')