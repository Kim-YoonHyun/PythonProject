# set 을 불러와서 갯수 늘리기
# translate 적용
# up sampling
# rotation
# random sampling

# _3
# target 갯수 증가
# Max, Min 구분 해서 계산

# _4
# 증가를 0, 1 로 했을때 되도록 수정

# _5
# 정리 word 파일과 맞추기
# Functions import 방식 변경

# _6
# Functions 을 직접 불러오는 방식으로 다시 되돌림
# 코드 리뉴얼
# _6_1
# class 사용해서 코드 간소화

# _7
# class 활용하여 target, bone1, bone2, skin 각 데이터별 정보 분리,
# up, random sampling 및 translate, rotation 적용 여부 및 각 데이터 정보를 전부 class 에 저장
# dill 추가 (class 인스턴스 객체 저장용)

# _8
# rotation 완료
# pandas 를 통해 data 저장

# _1_2_1
# class module 저장
# mutil target 생성 추가 중
# _1_3 
# 함수, 변수, 코드 순서로 코드 구조 변경
# Augmentation_1_3 에 따라 num_of_data, num_of_data_points 관련 코드 전부 삭제

import numpy as np
import os
# import functions_my_1_2_1 as fmy
import random
import dill  # _7
import pandas as pd  # _8
import Augmentation_1_3 as Aug
import warnings
import vtk
import math
warnings.filterwarnings(action='ignore')


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


def disc_to_target(stl_vertices_list):
    stl_vertices_array_center = np.average(np.array(stl_vertices_list), axis=0)
    min_y_coordinates = np.min(np.array(stl_vertices_list)[:, 1:2], axis=0)[0]
    while stl_vertices_array_center[1] > min_y_coordinates - 1:
        stl_vertices_array_center[1] -= 1
    target_vertex_array = np.expand_dims(stl_vertices_array_center, axis=0)
    return target_vertex_array.tolist()


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


def print_data_information(*args):
    for data in args:
        print(data, '\n')


def data_concatenate(*args):
    concat_data = []
    for arg in args:
        concat_data.append(arg)
    concat_data = np.concatenate(concat_data, axis=1)
    return concat_data


def main():
    global first_stl_set_num, last_stl_set_num, up_num_of_skin_points, up_num_of_bone1_points, up_num_of_bone2_points, num_of_rand_sam
    global num_of_trans, trans_x, trans_y, trans_z, num_of_rot, rot_x, rot_y, rot_z, mtar_x, mtar_y, mtar_z, num_of_multi_target

    # set 번호 범위 지정
    stl_set_path = f'stl/'

    # 각 데이터별 class 세팅
    target_data = Aug.DataInformation('target')
    bone1_data = Aug.DataInformation('bone1')
    bone2_data = Aug.DataInformation('bone2')
    skin_data = Aug.DataInformation('skin')
    Aug.DataInformation.aaa = 1

    # <초기 데이터 불러오기> ------------------------------------------------------------
    for stl_set_num in range(first_stl_set_num, last_stl_set_num + 1):
        file_list = os.listdir(f'{stl_set_path}/set{stl_set_num}')
        bone_flag = 1
        for file_name in file_list:
            if '.stl' in file_name:
                stl_name = file_name.split('.')[-2]
                stl_vertices_list, num_of_stl_points = read_stl(f'{stl_set_path}/set{stl_set_num}', stl_name)
                if stl_name[0] == 'D':
                    target_data.data_vertices_list.append(disc_to_target(stl_vertices_list))
                elif stl_name[0] == 'L' and bone_flag == 1:
                    bone1_data.data_vertices_list.append(stl_vertices_list)
                    bone_flag += 1
                elif stl_name[0] == 'L' and bone_flag == 2:
                    bone2_data.data_vertices_list.append(stl_vertices_list)
                elif stl_name[0] == 'S':
                    skin_data.data_vertices_list.append(stl_vertices_list)
                else:
                    continue
    target_data.up_sam_status = 'disc to target'
    target_data.array_size = np.array(target_data.data_vertices_list)

    print(f'{"target data is updated":>22s}')
    print(f"{'bone1 data is updated':>22s}")
    print(f"{'bone2 data is updated':>22s}")
    print(f"{'skin data is updated':>22s}", '\n')

    print_data_information(target_data, bone1_data, bone2_data, skin_data)

    # < up sampling > -----------------------------------------------------------
    print('up sampling...')
    bone1_data.up_sampling(up_num_of_bone1_points)
    bone2_data.up_sampling(up_num_of_bone2_points)
    skin_data.up_sampling(up_num_of_skin_points)

    # <random sampling> -----------------------------------------------------------
    print('random sampling...')
    target_data.data_vertices_list = [
        np.tile(np.array(target_data.data_vertices_list)[i], [num_of_rand_sam, 1, 1])[j].tolist()
        for i in range(len(target_data.data_vertices_list))
        for j in range(num_of_rand_sam)
    ]
    target_data.rand_sam_status = f'copyX{num_of_rand_sam}'
    bone1_data.random_sampling(up_num_of_bone1_points, num_of_rand_sam)
    bone2_data.random_sampling(up_num_of_bone2_points, num_of_rand_sam)
    skin_data.random_sampling(up_num_of_skin_points, num_of_rand_sam)

    print_data_information(target_data, bone1_data, bone2_data, skin_data)

    # <translate> -----------------------------------------------------------
    if num_of_trans > 1:
        print('translate...')
        # make trans offset
        target_trans_offset = make_trans_offset(len(target_data.data_vertices_list), num_of_trans, trans_x, trans_y, trans_z)
        bone_trans_offset = make_trans_offset(len(bone1_data.data_vertices_list), num_of_trans, trans_x, trans_y, trans_z)
        skin_trans_offset = make_trans_offset(len(skin_data.data_vertices_list), num_of_trans, trans_x, trans_y, trans_z)

        # translate
        target_data.translate(target_trans_offset, num_of_trans)
        bone1_data.translate(bone_trans_offset, num_of_trans)
        bone2_data.translate(bone_trans_offset, num_of_trans)
        skin_data.translate(skin_trans_offset, num_of_trans)
    else:
        print(f'not translate')

    # < rotation > ------------------------------------------------
    if num_of_rot > 1:
        print('rotation...')
        # rotation matrix
        target_rot_matrix = make_rot_matrices(len(target_data.data_vertices_list), num_of_rot, rot_x, rot_y, rot_z)
        bone_rot_matrix = make_rot_matrices(len(bone1_data.data_vertices_list), num_of_rot, rot_x, rot_y, rot_z)
        skin_rot_matrix = make_rot_matrices(len(skin_data.data_vertices_list), num_of_rot, rot_x, rot_y, rot_z)

        # rotation
        target_data.rotation(num_of_rot, target_rot_matrix)
        bone1_data.rotation(num_of_rot, bone_rot_matrix)
        bone2_data.rotation(num_of_rot, bone_rot_matrix)
        skin_data.rotation(num_of_rot, skin_rot_matrix)
    else:
        print('not rotation')

    # <데이터 병합(target 제외)>-------------------------------------------------------
    input_data_except_target = data_concatenate(bone1_data.data_vertices_list,
                                                bone2_data.data_vertices_list,
                                                skin_data.data_vertices_list)

    # <multi target>----------------------------------------------------
    if num_of_multi_target > 1:
        mul_tar_offset = make_trans_offset(len(target_data.data_vertices_list), num_of_multi_target, mtar_x, mtar_y, mtar_z)
        target_data.translate(mul_tar_offset, num_of_multi_target)
        # 각 input data에 multi target 적용
        multi_data_list = []
        for i in range(len(bone1_data.data_vertices_list)):
            tile_all_data_except_target = np.tile(input_data_except_target[i], [num_of_multi_target, 1, 1])
            multi_tar = np.array(target_data.data_vertices_list)[
                        0 + num_of_multi_target * i:num_of_multi_target * (1 + i), :, :]
            multi_data = np.concatenate((tile_all_data_except_target, multi_tar), axis=1)
            multi_data_list.append(multi_data)
        input_data = np.concatenate(multi_data_list, axis=0)
    else:
        input_data = data_concatenate(input_data_except_target, target_data.data_vertices_list)

    print_data_information(target_data, bone1_data, bone2_data, skin_data)
    print(f'input data size: {input_data.shape}')

    all_data = [target_data, bone1_data, bone2_data, skin_data]
    all_data_df = pd.DataFrame([target_data.__dict__, bone1_data.__dict__, bone2_data.__dict__, skin_data.__dict__])
    all_data_df.drop(columns='data_vertices_list', inplace=True)

    # <각종 데이터 저장>--------------------------------------------------------------------
    # 저장 폴더 만들기
    num_of_data_file = len(os.walk('./data/').__next__()[1])
    data_save_path = f'./data/data{num_of_data_file + 1}'
    if not os.path.isdir(f'{data_save_path}'):  # 해당 경로 내에 폴더가 없으면,
        os.makedirs(f'{data_save_path}')  # 그 폴더를 만들기

    # class 객체 저장
    with open(f'{data_save_path}/data_information.pkl', 'wb') as file:
        dill.dump(all_data, file)

    # csv 저장
    all_data_df.to_csv(f'{data_save_path}/data_information.csv', encoding='utf-8-sig')

    # npy 저장
    np.save(f'{data_save_path}/input_data', input_data)

    # txt 저장
    with open(f'{data_save_path}/stl set range = {first_stl_set_num} ~ {last_stl_set_num}.txt', 'w',
              encoding='utf8') as file:
        file.write(f'이 데이터는 stl set{first_stl_set_num} 부터 {last_stl_set_num} 까지,\n'
                   f'총 {last_stl_set_num - first_stl_set_num + 1}개 로 만들어진 data 입니다.')
    exit()


# global parameter
# A. stl 파일 불러오기. 현재 사용 중인 set 은 32번 부터이다.
first_stl_set_num = 32  # 불러올 set 의 시작 번호
last_stl_set_num = 33  # 불러올 set 의 마지막 번호

# C. up sampling
up_num_of_skin_points = 2700  # 증가시킬 skin 의 point 갯수
up_num_of_bone1_points = 1500  # 증가시킬 Bone1 의 point 갯수
up_num_of_bone2_points = 1500  # 증가시킬 Bone2 의 point 갯수

# random sampling
num_of_rand_sam = 3

# B. Augmentation: translate
num_of_trans = 3  # translate 를 통해 증가시킬 배수
trans_x = [-3.0, 3.0]  # x 축에 대한 최대 translate 범위
trans_y = [-3.0, 3.0]  # y 축에 대한 최대 translate 범위
trans_z = [-3.0, 3.0]  # z 축에 대한 최대 translate 범위

# B. Augmentation: rotation
num_of_rot = 3  # rotation 을 통해 증가시킬 배수
rot_x = [-5.0, 5.0]  # x 축에 대한 최대 rotation 범위
rot_y = [-5.0, 5.0]  # y 축에 대한 최대 rotation 범위
rot_z = [-5.0, 5.0]  # z 축에 대한 최대 rotation 범위

# D. 랜덤 타겟 생성
mtar_x = [-18.0, 18.0]
mtar_y = [-3.0, 7.0]
mtar_z = [-2.0, 2.0]
num_of_multi_target = 5


if __name__ == '__main__':
    main()