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
# _1_3_1
# 전역변수를 전부 대문자로 변경
# all_data 리스트를 초기에 만들어서 진행. []의 경우 업데이트가 상호적용되는 것을 확인 가능하였음
# 현재버전값 저장 추가
# 버전에 따른 저장방식 설정 추가
# Augmentation_1_3_1 을 병합
# version.ver 파일 저장 추가 20210903

import numpy as np
import os
import random
import dill  # _7
import pandas as pd  # _8
import warnings
import vtk
import math
import re  # _1_3_1
import copy
warnings.filterwarnings(action='ignore')


class DataInformation:
    num = 0

    def __init__(self, data_name):
        self.data_name = data_name
        self.data_vertices_list = []
        self.up_sam_status = ''
        self.rand_sam_status = ''
        self.trans_status = ''
        self.rot_status = ''

        DataInformation.num += 1

    def __str__(self):
        return f'<{self.data_name}>\n' \
               f'array size: {np.array(self.data_vertices_list).shape}\n' \
               f'up sample:{self.up_sam_status}\n' \
               f'random sample:{self.rand_sam_status}\n' \
               f'translate:{self.trans_status}\n' \
               f'rotation:{self.rot_status}'

    def up_sampling(self, up_num_of_stl_points):
        """
        데이터 vertices의 갯수를 지정한 갯수만큼 추가 생성하는 함수
        :param up_num_of_stl_points: 추가생성을 통해 달성할 목표 갯수
        :return:
        """
        for data_index, data in enumerate(self.data_vertices_list):
            print(f'data{data_index}: {np.array(data).shape} --->', end=' ')
            # 추출용 copy vertices 생성
            vertices_list_copy = copy.deepcopy(data)
            num_of_data_points = np.array(data).shape[0]
            for _ in range(up_num_of_stl_points - num_of_data_points):
                # 랜덤으로 vertex 하나 추출
                rand_vertex = vertices_list_copy.pop(random.randint(0, len(vertices_list_copy) - 1))
                # 추출된 vertex 와 나머지 모든 vertex 간의 거리 계산
                distance_of_rand_vertex_to_rest_vertices = euclidean_distance(rand_vertex, vertices_list_copy)
                # 추출된 vertex 와 가장 가까운 vertex 선택
                nearest_vertex = vertices_list_copy[np.argmin(distance_of_rand_vertex_to_rest_vertices)]
                # 추출된 vertex 와 가장 가까운 vertex 의 중간지점에 새로운 vertex 생성
                middle_vertex = list(np.average((np.array(rand_vertex), np.array(nearest_vertex)), axis=0))
                # 생성된 새로운 vertex 를 기존 vertices 에 추가
                data.append(middle_vertex)
            print(np.array(data).shape)
        print()
        self.up_sam_status = up_num_of_stl_points

    def random_sampling(self, up_num_of_stl_points, num_of_rand_sam):
        """
        :param up_num_of_stl_points:
        :param num_of_rand_sam:
        :return:
        """
        print(f'<{self.data_name}>')
        ran_sam_vertices_list = []
        for data_index, data in enumerate(self.data_vertices_list):
            print(f'data{data_index}: {np.array(data).shape} --->', end=' ')
            for _ in range(num_of_rand_sam):
                temp_list = []
                for _ in range(int(up_num_of_stl_points / num_of_rand_sam)):
                    # 랜덤으로 vertex 하나 선택 후 추출
                    rand_index = random.randint(0, len(data) - 1)
                    temp_list.append(data.pop(rand_index))
                # 목표치 만큼 추출된 vertices list 를 저장
                ran_sam_vertices_list.append(temp_list)
            print(f'{np.array(temp_list).shape} X {num_of_rand_sam}')
        print()
        self.data_vertices_list = ran_sam_vertices_list
        self.rand_sam_status += f' rand /{num_of_rand_sam}'

    def translate(self, xyz_offset, num_of_trans, multi):
        """
        :param xyz_offset:
        :param num_of_trans:
        :return:
        """
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        num_of_data = np.array(self.data_vertices_list).shape[0]
        xyz_offset = np.expand_dims(np.array(xyz_offset), axis=-2)
        temp_list = []
        for i in range(num_of_data):
            temp_list.append(np.add(self.data_vertices_list[i], xyz_offset[i]).tolist())

        temp_list = [np.add(self.data_vertices_list[i], xyz_offset[i]).tolist() for i in range(num_of_data)]
        result_vertices = np.concatenate(temp_list, axis=0).tolist()
        print(f'{np.array(result_vertices).shape}')
        print()
        self.data_vertices_list = result_vertices
        if multi is not None:
            self.trans_status += f' multi X{num_of_trans}'
        else:
            self.trans_status += f' trans X{num_of_trans}'

    def rotation(self, xyz_rot_matrices, num_of_rot):
        """
        :param xyz_rot_matrices:
        :return:
        """

        num_of_data = np.array(self.data_vertices_list).shape[0]
        center_vertex = np.average(np.array(self.data_vertices_list), axis=1)
        temp_list = []
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        for i in range(num_of_data):
            # 원점으로 옮기기
            trans_to_zero_coordinate = np.subtract(np.array(self.data_vertices_list)[i], center_vertex[i])

            # rotation
            rot_vertices = np.dot(trans_to_zero_coordinate, xyz_rot_matrices[i])
            rot_vertices_expanded = np.expand_dims(rot_vertices, axis=-2)
            result_vertices_zero = np.concatenate((rot_vertices_expanded), axis=1)

            # 원래 위치로 되돌리기
            trans_to_origin_coordinate = np.add(result_vertices_zero, center_vertex[i])

            # list 에 넣기
            temp_list.append(trans_to_origin_coordinate)

        all_result_vertices = np.concatenate(temp_list, axis=0)
        print(f'{all_result_vertices.shape}')
        print()
        self.data_vertices_list = all_result_vertices.tolist()
        self.rot_status += f' rot X{num_of_rot}'

def euclidean_distance(start, end):
    """
    function of calculate the euclidean distance from start to end
    :param start: start vertex
    :param end: end vertex
    :return: euclidean distance
    """

    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
    return value


def print_data_information(*args):
    for data in args:
        print(data, '\n')


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


def data_vertices_extraction(path, start_num, last_num):
    data_vertices_collection = []
    for stl_set_num in range(start_num, last_num + 1):
        file_list = os.listdir(f'{path}/set{stl_set_num}')
        for file_name in file_list:
            if '.stl' in file_name:
                stl_name = file_name.split('.')[-2]
                stl_vertices_list, _ = read_stl(f'{path}/set{stl_set_num}', stl_name)
                data_vertices_collection.append(stl_vertices_list)
    return data_vertices_collection


def disc_to_target(stl_vertices_list):
    stl_vertices_array_center = np.average(np.array(stl_vertices_list), axis=0)
    min_y_coordinates = np.min(np.array(stl_vertices_list)[:, 1:2], axis=0)[0]
    while stl_vertices_array_center[1] > min_y_coordinates - 1:
        stl_vertices_array_center[1] -= 1
    target_vertex_array = np.expand_dims(stl_vertices_array_center, axis=0)
    return target_vertex_array.tolist()


def make_trans_offset(num_of_data, NUM_OF_TRANS, TRANS_X, TRANS_Y, TRANS_Z):
    xyz_offset = [
        [
            [round(random.uniform(TRANS_X[0], TRANS_X[1]), 5),
             round(random.uniform(TRANS_Y[0], TRANS_Y[1]), 5),
             round(random.uniform(TRANS_Z[0], TRANS_Z[1]), 5)] for _ in range(NUM_OF_TRANS)
        ] for i in range(num_of_data)
    ]
    # xyz_offset = np.expand_dims(np.array(xyz_offset), axis=2)
    return xyz_offset
def make_rot_matrices(num_of_data, NUM_OF_ROT, ROT_X, ROT_Y, ROT_Z):
    """
    point cloud에 회전을 적용하기 위한 matrix를 계산하는 함수.
    :param num_of_data: 회전을 적용할 데이터의 갯수
    :param NUM_OF_ROT: 회전을 통해 늘릴 데이터 배수
    :param ROT_X: x 축으로 회전을 적용할 범위
    :param ROT_Y: y 축으로 회전을 적용할 범위
    :param ROT_Z: z 축으로 회전을 적용할 범위
    :return: (num_of_data, NUM_OF_ROT, 3, 3)
    """
    all_rot_matrices = []
    for j in range(num_of_data):
        xyz_rot_matrix = []
        for i in range(NUM_OF_ROT):
            x_angle = random.uniform(ROT_X[0], ROT_X[1]) * (math.pi / 180)
            y_angle = random.uniform(ROT_Y[0], ROT_Y[1]) * (math.pi / 180)
            z_angle = random.uniform(ROT_Z[0], ROT_Z[1]) * (math.pi / 180)
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


def data_concatenate(*args):
    concat_data = []
    for arg in args:
        concat_data.append(arg)
    concat_data = np.concatenate(concat_data, axis=1)
    return concat_data


def extract_version():
    name = os.path.basename(__file__)
    name = re.sub('[_.]', ' ', name).split()
    ver = ''
    for word in name:
        try:
            ver_word = int(word)
            ver += f'{ver_word}'
        except:
            pass
    return '.'.join(ver)


def main():
    global FIRST_STL_SET_NUM, LAST_STL_SET_NUM, UP_NUM_OF_SKIN_POINTS, UP_NUM_OF_BONE1_POINTS, UP_NUM_OF_BONE2_POINTS, NUM_OF_RAND_NUM
    global NUM_OF_TRANS, TRANS_X, TRANS_Y, TRANS_Z, NUM_OF_ROT, ROT_X, ROT_Y, ROT_Z, MTAR_X, MTAR_Y, MTAR_Z, NUM_OF_MULTI_TARGET
    global CLASS_N_DF, OTHER

    # 각 데이터별 class 세팅
    target_data = DataInformation('target')
    bone1_data = DataInformation('bone1')
    bone2_data = DataInformation('bone2')
    skin_data = DataInformation('skin')
    all_data = [target_data, bone1_data, bone2_data, skin_data]
    print_data_information(target_data, bone1_data, bone2_data, skin_data)

    # <초기 데이터 불러오기> ------------------------------------------------------------
    data_vertices_collection = data_vertices_extraction(f'stl/', FIRST_STL_SET_NUM, LAST_STL_SET_NUM)
    print('data update...')
    for idx, data in enumerate(data_vertices_collection):
        if idx % 4 == 0:
            data = disc_to_target(data)
        all_data[idx % 4].data_vertices_list.append(data)
    print_data_information(target_data, bone1_data, bone2_data, skin_data)

    # < up sampling > -----------------------------------------------------------
    print('up sampling...')
    bone1_data.up_sampling(UP_NUM_OF_BONE1_POINTS)
    bone2_data.up_sampling(UP_NUM_OF_BONE2_POINTS)
    skin_data.up_sampling(UP_NUM_OF_SKIN_POINTS)

    # <random sampling> -----------------------------------------------------------
    print('random sampling...')
    target_data.data_vertices_list = [
        np.tile(np.array(target_data.data_vertices_list)[i], [NUM_OF_RAND_NUM, 1, 1])[j].tolist()
        for i in range(len(target_data.data_vertices_list))
        for j in range(NUM_OF_RAND_NUM)
    ]
    target_data.rand_sam_status = f'copy X{NUM_OF_RAND_NUM}'
    bone1_data.random_sampling(UP_NUM_OF_BONE1_POINTS, NUM_OF_RAND_NUM)
    bone2_data.random_sampling(UP_NUM_OF_BONE2_POINTS, NUM_OF_RAND_NUM)
    skin_data.random_sampling(UP_NUM_OF_SKIN_POINTS, NUM_OF_RAND_NUM)

    # <translate> -----------------------------------------------------------
    if NUM_OF_TRANS > 1:
        print('translate...')
        # make trans offset
        target_trans_offset = make_trans_offset(len(target_data.data_vertices_list), NUM_OF_TRANS, TRANS_X, TRANS_Y, TRANS_Z)
        bone_trans_offset = make_trans_offset(len(bone1_data.data_vertices_list), NUM_OF_TRANS, TRANS_X, TRANS_Y, TRANS_Z)
        skin_trans_offset = make_trans_offset(len(skin_data.data_vertices_list), NUM_OF_TRANS, TRANS_X, TRANS_Y, TRANS_Z)

        # translate
        target_data.translate(target_trans_offset, NUM_OF_TRANS, None)
        bone1_data.translate(bone_trans_offset, NUM_OF_TRANS, None)
        bone2_data.translate(bone_trans_offset, NUM_OF_TRANS, None)
        skin_data.translate(skin_trans_offset, NUM_OF_TRANS, None)
    else:
        print(f'not translate')

    # < rotation > ------------------------------------------------
    if NUM_OF_ROT > 1:
        print('rotation...')
        # rotation matrix
        target_rot_matrix = make_rot_matrices(len(target_data.data_vertices_list), NUM_OF_ROT, ROT_X, ROT_Y, ROT_Z)
        bone_rot_matrix = make_rot_matrices(len(bone1_data.data_vertices_list), NUM_OF_ROT, ROT_X, ROT_Y, ROT_Z)
        skin_rot_matrix = make_rot_matrices(len(skin_data.data_vertices_list), NUM_OF_ROT, ROT_X, ROT_Y, ROT_Z)

        # rotation
        target_data.rotation(target_rot_matrix, NUM_OF_ROT)
        bone1_data.rotation(bone_rot_matrix, NUM_OF_ROT)
        bone2_data.rotation(bone_rot_matrix, NUM_OF_ROT)
        skin_data.rotation(skin_rot_matrix, NUM_OF_ROT)
    else:
        print('not rotation')

    # <데이터 병합(target 제외)>-------------------------------------------------------
    input_data_except_target = data_concatenate(bone1_data.data_vertices_list,
                                                bone2_data.data_vertices_list,
                                                skin_data.data_vertices_list)

    # <multi target>----------------------------------------------------
    if NUM_OF_MULTI_TARGET > 1:
        mul_tar_offset = make_trans_offset(len(target_data.data_vertices_list), NUM_OF_MULTI_TARGET, MTAR_X, MTAR_Y, MTAR_Z)
        target_data.translate(mul_tar_offset, NUM_OF_MULTI_TARGET, 'on')

        np.tile(bone1_data.data_vertices_list, [NUM_OF_MULTI_TARGET, 1, 1])
        # 각 input data에 multi target 적용
        multi_data_list = []
        print(len(bone1_data.data_vertices_list))
        exit()
        for i in range(len(bone1_data.data_vertices_list)):
            tile_all_data_except_target = np.tile(input_data_except_target[i], [NUM_OF_MULTI_TARGET, 1, 1])
            multi_tar = np.array(target_data.data_vertices_list)[
                        (NUM_OF_MULTI_TARGET * i):(NUM_OF_MULTI_TARGET * (i+1)), :, :]
            multi_data = np.concatenate((tile_all_data_except_target, multi_tar), axis=1)
            multi_data_list.append(multi_data)
        input_data = np.concatenate(multi_data_list, axis=0)
    else:
        input_data = data_concatenate(bone1_data.data_vertices_list, bone2_data.data_vertices_list,
                                      skin_data.data_vertices_list, target_data.data_vertices_list)

    print_data_information(target_data, bone1_data, bone2_data, skin_data)
    print(f'input data size: {input_data.shape}')

    # <각종 데이터 저장>--------------------------------------------------------------------
    # 현재 저장 데이터: pkl, csv, txt
    # 저장 폴더 만들기
    num_of_data_file = len(os.walk('./data/').__next__()[1])
    data_save_path = f'./data/data{num_of_data_file + 1}'
    if not os.path.isdir(f'{data_save_path}'):  # 해당 경로 내에 폴더가 없으면,
        os.makedirs(f'{data_save_path}')  # 그 폴더를 만들기

    # class 객체 저장 : 각 데이터별 인스턴스 정보
    with open(f'{data_save_path}/{CLASS_N_DF}.pkl', 'wb') as file:
        dill.dump(all_data, file)

    # csv 저장 : 각 데이터별 인스턴스 정보
    all_data_df = pd.DataFrame([target_data.__dict__, bone1_data.__dict__, bone2_data.__dict__, skin_data.__dict__])
    all_data_df.rename(columns={'data_vertices_list':'array_size'}, inplace=True)
    all_data_df['array_size'] = all_data_df['array_size'].apply(lambda x: np.array(x).shape)
    all_data_df.to_csv(f'{data_save_path}/{CLASS_N_DF}.csv', encoding='utf-8-sig')

    # npy 저장 : 학습용 input data
    np.save(f'{data_save_path}/input_data', input_data)

    # txt 저장 : 그외 기타 정보
    with open(f'{data_save_path}/{OTHER}.txt', 'w',
              encoding='utf8') as file:
        file.write(f"'stl set range': '{FIRST_STL_SET_NUM} ~ {LAST_STL_SET_NUM}'\n")

    # ver 저장: 현재 코드 버전
    with open(f'{data_save_path}/{extract_version()}.ver', 'w') as file:
        file.close()
    exit()


# global parameter
# A. stl 파일 불러오기. 현재 사용 중인 set 은 32번 부터이다.
FIRST_STL_SET_NUM = 32  # 불러올 set 의 시작 번호
LAST_STL_SET_NUM = 33  # 불러올 set 의 마지막 번호

# C. up sampling
UP_NUM_OF_SKIN_POINTS = 2700  # 증가시킬 skin 의 point 갯수
UP_NUM_OF_BONE1_POINTS = 1500  # 증가시킬 Bone1 의 point 갯수
UP_NUM_OF_BONE2_POINTS = 1500  # 증가시킬 Bone2 의 point 갯수

# random sampling
NUM_OF_RAND_NUM = 3

# B. Augmentation: translate
NUM_OF_TRANS = 3  # translate 를 통해 증가시킬 배수
TRANS_X = [-3.0, 3.0]  # x 축에 대한 최대 translate 범위
TRANS_Y = [-3.0, 3.0]  # y 축에 대한 최대 translate 범위
TRANS_Z = [-3.0, 3.0]  # z 축에 대한 최대 translate 범위

# B. Augmentation: rotation
NUM_OF_ROT = 3  # rotation 을 통해 증가시킬 배수
ROT_X = [-5.0, 5.0]  # x 축에 대한 최대 rotation 범위
ROT_Y = [-5.0, 5.0]  # y 축에 대한 최대 rotation 범위
ROT_Z = [-5.0, 5.0]  # z 축에 대한 최대 rotation 범위

# D. 랜덤 타겟 생성
MTAR_X = [-18.0, 18.0]
MTAR_Y = [-3.0, 7.0]
MTAR_Z = [-2.0, 2.0]
NUM_OF_MULTI_TARGET = 5

# 저장 명칭
CLASS_N_DF = 'data_information'
OTHER = 'other_information'

if __name__ == '__main__':
    main()