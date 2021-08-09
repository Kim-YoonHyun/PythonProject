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

import numpy as np
import os
import functions_my
import copy
import random
import math
import vtk
import dill  # _7

# #### basic parameter
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
tar_x = [-18.0, 18.0]
tar_y = [-3.0, 7.0]
tar_z = [-2.0, 2.0]
num_of_target = 2


# ############################## Make dict #############################################################################
# set 번호 범위 지정
stl_set_order_range = list(range(first_stl_set_num, last_stl_set_num + 1))
stl_set_path = f'stl/'

class DataInformation:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_vertices_list = []
        self.num_of_data = 0
        self.num_of_data_points = []
        self.up_sam_status = None
        self.rand_sam_status = None
        self.trans_status = None
        self.rot_status = None

    def __str__(self):
        return f'<{self.data_name}>\n' \
               f'number of data: {self.num_of_data}\n' \
               f'number of data points: {self.num_of_data_points}\n' \
               f'array size: {np.array(self.data_vertices_list).shape}\n' \
               f'up sample: {self.up_sam_status}\n' \
               f'random sample: {self.rand_sam_status}\n' \
               f'translate: {self.trans_status}\n' \
               f'rotation: {self.rot_status}'

    def up_sampling(self, up_num_of_stl_points):
        """
        :param up_num_of_stl_points:
        :return:
        """
        for data_index in range(self.num_of_data):
            temp_vertices_list = copy.deepcopy(self.data_vertices_list[data_index])
            print(f'data{data_index}: {np.array(temp_vertices_list).shape} --->', end=' ')
            for _ in range(up_num_of_stl_points - self.num_of_data_points[data_index]):
                rand_vertex = temp_vertices_list.pop(random.randint(0, len(temp_vertices_list) - 1))
                distance_of_rand_vertex_to_rest_vertices = functions_my.euclidean_distance(rand_vertex, temp_vertices_list)
                nearest_vertex = temp_vertices_list[np.argmin(distance_of_rand_vertex_to_rest_vertices)]
                middle_vertex = list(np.average((np.array(rand_vertex), np.array(nearest_vertex)), axis=0))
                self.data_vertices_list[data_index].append(middle_vertex)
            print(np.array(self.data_vertices_list[data_index]).shape)
        self.num_of_data_points = up_num_of_stl_points
        self.up_sam_status = up_num_of_stl_points
        print()

    def random_sampling(self, up_num_of_stl_points, num_of_rand_sam):
        """

        :param up_num_of_stl_points:
        :param num_of_rand_sam:
        :return:
        """
        print(f'<{self.data_name}>')
        ran_sam_vertices_list = []
        for data_index in range(self.num_of_data):
            print(f'data{data_index}: {np.array(self.data_vertices_list[data_index]).shape} --->', end=' ')
            for _ in range(num_of_rand_sam):
                temp_list = []
                for _ in range(int(up_num_of_stl_points / num_of_rand_sam)):
                    rand_index = random.randint(0, len(self.data_vertices_list[data_index]) - 1)
                    temp_list.append(self.data_vertices_list[data_index].pop(rand_index))
                ran_sam_vertices_list.append(temp_list)
            print(f'{np.array(temp_list).shape} X {num_of_rand_sam}')
        self.data_vertices_list = ran_sam_vertices_list
        self.num_of_data *= num_of_rand_sam
        self.num_of_data_points = int(up_num_of_stl_points / num_of_rand_sam)
        self.rand_sam_status = f'/{num_of_rand_sam}'
        print()

    def translate(self, xyz_offset, num_of_trans):
        """

        :param xyz_offset:
        :param data_vertices_list:
        :return:
        """
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        for data_index in range(self.num_of_data):
            temp_list = [
                np.add(np.array(self.data_vertices_list[data_index]), xyz_offset[i]).tolist()
                for i in range(len(xyz_offset))
            ]

        self.data_vertices_list = np.concatenate(temp_list, axis=0).tolist()
        print(f'{np.array(self.data_vertices_list).shape}')
        self.num_of_data *= num_of_trans
        self.num_of_data_points = np.array(self.data_vertices_list).shape[1]
        self.trans_status = f'trans X {num_of_trans}'
        print()

target_data = DataInformation('target')
bone1_data = DataInformation('bone1')
bone2_data = DataInformation('bone2')
skin_data = DataInformation('skin')

for stl_set_num in range(first_stl_set_num, last_stl_set_num + 1):
    file_list = os.listdir(f'{stl_set_path}/set{stl_set_num}')
    bone_flag = 1
    for file_name in file_list:
        if '.stl' in file_name:
            stl_name = file_name.split('.')[-2]
            stl_vertices_list, num_of_stl_points = functions_my.read_stl(f'{stl_set_path}/set{stl_set_num}', stl_name)

            # Disc 데이터를 target 데이터로 바꾸기
            if stl_name[0] == 'D':
                stl_vertices_array_center = np.average(np.array(stl_vertices_list), axis=0)
                min_y_coordinates = np.min(np.array(stl_vertices_list)[:, 1:2], axis=0)[0]
                while stl_vertices_array_center[1] > min_y_coordinates - 1:
                    stl_vertices_array_center[1] -= 1
                target_vertex_array = np.expand_dims(stl_vertices_array_center, axis=0)
                target_data.data_vertices_list.append(target_vertex_array.tolist())
                target_data.num_of_data += 1
            elif stl_name[0] == 'L' and bone_flag == 1:
                bone_flag += 1
                bone1_data.data_vertices_list.append(stl_vertices_list)
                bone1_data.num_of_data_points.append(num_of_stl_points)
                bone1_data.num_of_data += 1
            elif stl_name[0] == 'L' and bone_flag == 2:
                bone2_data.data_vertices_list.append(stl_vertices_list)
                bone2_data.num_of_data_points.append(num_of_stl_points)
                bone2_data.num_of_data += 1
            elif stl_name[0] == 'S':
                skin_data.data_vertices_list.append(stl_vertices_list)
                skin_data.num_of_data_points.append(num_of_stl_points)
                skin_data.num_of_data += 1
            else:
                continue

target_data.num_of_data_points = 1
target_data.up_sam_status = 'disc to target'
print(f'{"target data is updated":>22s}')
print(f"{'bone1 data is updated':>22s}")
print(f"{'bone2 data is updated':>22s}")
print(f"{'skin data is updated':>22s}", '\n')

print(target_data, '\n')
print(bone1_data, '\n')
print(bone2_data, '\n')
print(skin_data, '\n')

# < up sampling > -----------------------------------------------------------
print('up sampling...')
bone1_data.up_sampling(up_num_of_bone1_points)
bone2_data.up_sampling(up_num_of_bone2_points)
skin_data.up_sampling(up_num_of_skin_points)

# <random sampling> -----------------------------------------------------------
print('random sampling...')
target_data.data_vertices_list = [np.tile(np.array(target_data.data_vertices_list)[i], [num_of_rand_sam, 1, 1])[j].tolist()
                                  for i in range(len(target_data.data_vertices_list))
                                  for j in range(num_of_rand_sam)
                                  ]
target_data.num_of_data *= num_of_rand_sam
target_data.rand_sam_status = f'copy X {num_of_rand_sam}'
bone1_data.random_sampling(up_num_of_bone1_points, num_of_rand_sam)
bone2_data.random_sampling(up_num_of_bone2_points, num_of_rand_sam)
skin_data.random_sampling(up_num_of_skin_points, num_of_rand_sam)


# <translate> -----------------------------------------------------------
if num_of_trans > 1:
    print('translate...')
    xyz_offset = [
        [
            [round(random.uniform(trans_x[0], trans_x[1]), 5),
             round(random.uniform(trans_y[0], trans_y[1]), 5),
             round(random.uniform(trans_z[0], trans_z[1]), 5)] for _ in range(num_of_trans)
        ] for i in range(target_data.num_of_data)
    ]
    xyz_offset = np.expand_dims(np.array(xyz_offset), axis=2)
    target_data.translate(xyz_offset, num_of_trans)
    bone1_data.translate(xyz_offset, num_of_trans)
    bone2_data.translate(xyz_offset, num_of_trans)
    skin_data.translate(xyz_offset, num_of_trans)
else:
    print(f'not translate')

# < rotation > ------------------------------------------------

print(target_data, '\n')
print(bone1_data, '\n')
print(bone2_data, '\n')
print(skin_data, '\n')

input_data = [target_data, bone1_data, bone2_data, skin_data]
with open('input_data.pkl', 'wb') as file:
    dill.dump(input_data, file)

#     temp_target_list, temp_bone1_list, temp_bone2_list, temp_skin_list = [], [], [], []
#     for i in range(len(ran_up_all_bone1_vertices_list)):
#
#         xyz_offset = [[round(random.uniform(trans_x[0], trans_x[1]), 5),
#                        round(random.uniform(trans_y[0], trans_y[1]), 5),
#                        round(random.uniform(trans_z[0], trans_z[1]), 5)]
#                       for _ in range(num_of_trans_increasing)]
#         xyz_offset = np.expand_dims(np.array(xyz_offset), axis=1)
#
#         temp_target_list.append(np.add(np.array(ran_all_target_vertex_list)[i], xyz_offset))
#         temp_bone1_list.append(np.add(np.array(ran_up_all_bone1_vertices_list[i]), xyz_offset))
#         temp_bone2_list.append(np.add(np.array(ran_up_all_bone2_vertices_list[i]), xyz_offset))
#         temp_skin_list.append(np.add(np.array(ran_up_all_skin_vertices_list[i]), xyz_offset))
#     trans_ran_all_target_vertex_list = np.concatenate((temp_target_list), axis=0).tolist()
#     trans_ran_up_all_bone1_vertices_list = list(np.concatenate((temp_bone1_list), axis=0))
#     trans_ran_up_all_bone2_vertices_list = list(np.concatenate((temp_bone2_list), axis=0))
#     trans_ran_up_all_skin_vertices_list = list(np.concatenate((temp_skin_list), axis=0))
#     print(f'translate X {num_of_trans_increasing}')
# else:
#     trans_ran_all_target_vertex_list = ran_all_target_vertex_list
#     trans_ran_up_all_bone1_vertices_list = ran_up_all_bone1_vertices_list
#     trans_ran_up_all_bone2_vertices_list = ran_up_all_bone2_vertices_list
#     trans_ran_up_all_skin_vertices_list = ran_up_all_skin_vertices_list
#     print(f'not translate')
#
# print(np.array(trans_ran_all_target_vertex_list).shape, end=' ')
# print(np.array(trans_ran_all_target_vertex_list), end=' ')
# print(np.array(trans_ran_up_all_bone1_vertices_list).shape, end=' ')
# print(np.array(trans_ran_up_all_bone2_vertices_list).shape, end=' ')
# print(np.array(trans_ran_up_all_skin_vertices_list).shape, '\n')


### rotation
def make_rotate_angle(number, x1, x2, y1, y2, z1, z2):
    x_degree = random.uniform(x1, x2) * (math.pi / 180)
    y_degree = random.uniform(y1, y2) * (math.pi / 180)
    z_degree = random.uniform(z1, z2) * (math.pi / 180)
    return x_degree, y_degree, z_degree

xd, yd, zd = make_rotate_angle(num_of_rot, rot_x[0], rot_x[1], rot_y[0], rot_y[1], rot_z[0], rot_z[1])

print(xd*180/math.pi)
print(yd*180/math.pi)
print(zd*180/math.pi)
## >>> rotation 진행중
def make_rotation_matrix(xyz_angle):
    x_matrix = np.expand_dims(np.array([
        [1.,                       0.,                      0.],
        [0.,   math.cos(xyz_angle[0]), -math.sin(xyz_angle[0])],
        [0.,   math.sin(xyz_angle[0]),  math.cos(xyz_angle[0])]]), axis=0)
    y_matrix = np.expand_dims(np.array([
        [math.cos(xyz_angle[1]),   0.,  math.sin(xyz_angle[1])],
        [0.,                       1.,                      0.],
        [-math.sin(xyz_angle[1]),  0.,  math.cos(xyz_angle[1])]]), axis=0)
    z_matrix = np.expand_dims(np.array([
        [math.cos(xyz_angle[2]), -math.sin(xyz_angle[2]),   0.],
        [math.sin(xyz_angle[2]),  math.cos(xyz_angle[2]),   0.],
        [0.,                       0.,                      1.]]), axis=0)
    return np.concatenate((x_matrix, y_matrix, z_matrix), axis=0)

temp_list = []
temp_list.append(make_rotation_matrix('x', xd))
temp_list.append(make_rotation_matrix('y', yd))
temp_list.append(make_rotation_matrix('z', zd))
print(np.array(temp_list))
print(np.array(temp_list).shape)

exit()




exit()
def make_all_rotation(number, x1, x2, y1, y2, z1, z2, array, center):
    answer_list = []
    x_angle, y_angle, z_angle = make_rotate_angle(number=number, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)
    for i in range(number):
        trans_to_zero = np.add(array, center)

        # x rotation
        rotation_mat = make_rotation_matrix(axis_select='x', angle=x_angle[i])
        rotation1 = np.dot(trans_to_zero, rotation_mat)

        # y rotation
        rotation_mat = make_rotation_matrix(axis_select='y', angle=y_angle[i])
        rotation2 = np.dot(rotation1, rotation_mat)

        # z rotation
        rotation_mat = make_rotation_matrix(axis_select='z', angle=z_angle[i])
        rotation3 = np.dot(rotation2, rotation_mat)

        back_to_origin = np.subtract(rotation3, center)

        answer_list.append(back_to_origin)
    return np.array(answer_list)
def make_all_rotation_mk2(number, x_angle, y_angle, z_angle, array):
    list = []
    center = -np.average(array, axis=0)

    for i in range(number):
        trans_to_zero = np.add(array, center)

        # x rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select= 'x', angle= x_angle[i])
        rotation1 = np.dot(trans_to_zero, rotation_mat)

        # y rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select='y', angle=y_angle[i])
        rotation2 = np.dot(rotation1, rotation_mat)

        # z rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select='z', angle=z_angle[i])
        rotation3 = np.dot(rotation2, rotation_mat)

        back_to_origin = np.subtract(rotation3, center)
        list.append(back_to_origin)
    return np.array(list)

# # translate
# if num_of_tar_increasing == 0:
#     # all_inc_up_Skin_array = all_origin_up_Skin_array
#     # all_inc_up_Bone1_array = all_origin_up_Bone1_array
#     # all_inc_up_Bone2_array = all_origin_up_Bone2_array
#     # all_inc_tar_array = all_origin_tar_array
#     a = 1
# else:
#     for i in range(len(set)):
#         # x_offset, y_offset, z_offset = Functions.make_offset_distance(trans_increasing_number, trans_x[0], trans_x[1],
#         #                                                               trans_y[0], trans_y[1], trans_z[0], trans_z[1])
#
#         # Skin
#         Skin_temp = Functions.make_all_offset(all_origin_up_Skin_array[i], x_offset, y_offset, z_offset,
#                                               trans_increasing_number)
#         Skin_dummy_array = np.concatenate((Skin_dummy_array, Skin_temp), axis=0)
#
#         # # Bone1
#         Bone1_temp = Functions.make_all_offset(all_origin_up_Bone1_array[i], x_offset, y_offset, z_offset,
#                                                trans_increasing_number)
#         Bone1_dummy_array = np.concatenate((Bone1_dummy_array, Bone1_temp), axis=0)
#
#         # Bone2
#         Bone2_temp = Functions.make_all_offset(all_origin_up_Bone2_array[i], x_offset, y_offset, z_offset,
#                                                trans_increasing_number)
#         Bone2_dummy_array = np.concatenate((Bone2_dummy_array, Bone2_temp), axis=0)
#
#         # tar
#         tar_temp = Functions.make_all_offset(all_origin_tar_array[i], x_offset, y_offset, z_offset,
#                                              trans_increasing_number)
#         tar_dummy_array = np.concatenate((tar_dummy_array, tar_temp), axis=0)
#
#         # dummy array 제거
#         all_inc_up_Skin_array = Skin_dummy_array[1:, :, :]
#         all_inc_up_Bone1_array = Bone1_dummy_array[1:, :, :]
#         all_inc_up_Bone2_array = Bone2_dummy_array[1:, :, :]
#         all_inc_tar_array = tar_dummy_array[1:, :, :]
#
# # =======================================================
# # random sampling
# Skin_rand_sam_num = int(skin_points/rand_sam_num)
# Bone1_rand_sam_num = int(Bone1_points/rand_sam_num)
# Bone2_rand_sam_num = int(Bone2_points/rand_sam_num)
# Skin_dummy_arrays = np.zeros([1, Skin_rand_sam_num, 3])
# Bone1_dummy_arrays = np.zeros([1, Bone1_rand_sam_num, 3])
# Bone2_dummy_arrays = np.zeros([1, Bone2_rand_sam_num, 3])
#
#
# all_inc_up_rand_Skin_array, _ = Functions.random_sampling(all_inc_up_Skin_array, Skin_dummy_arrays, Skin_rand_sam_num)
# all_inc_up_rand_Bone1_array, Bone1_tile = Functions.random_sampling(all_inc_up_Bone1_array, Bone1_dummy_arrays,
#                                                                     Bone1_rand_sam_num)
# all_inc_up_rand_Bone2_array, _ = Functions.random_sampling(all_inc_up_Bone2_array, Bone2_dummy_arrays,
#                                                            Bone2_rand_sam_num)
# all_inc_rand_tar_arrays = np.zeros([1, 1, 3])
#
# for i in range(all_inc_tar_array.shape[0]):
#     temp = np.tile(all_inc_tar_array[i], [Bone1_tile[i], 1, 1])
#     all_inc_rand_tar_arrays = np.concatenate((all_inc_rand_tar_arrays, temp), axis=0)
# all_inc_rand_tar_arrays = np.delete(all_inc_rand_tar_arrays, 0, axis=0)

# =======================================================
# rotation
# Skin
Skin_dummy_array = np.zeros([1, int(skin_points/rand_sam_num), 3])

if rot_increasing_number == 0:
    all_inc_up_rand_Skin_array = all_inc_up_rand_Skin_array
else:
    for i in range(all_inc_up_rand_Skin_array.shape[0]):
        x_angle, y_angle, z_angle = Functions.make_rotate_angle(rot_increasing_number,
                                                                rot_x[0], rot_x[1],
                                                                rot_y[0], rot_y[1],
                                                                rot_z[0], rot_z[1])
        Skin_temp = Functions_mk2.make_all_rotation_mk2(rot_increasing_number, x_angle, y_angle, z_angle,
                                                        all_inc_up_rand_Skin_array[i])
        Skin_dummy_array = np.concatenate((Skin_dummy_array, Skin_temp), axis=0)
    all_inc_up_rand_Skin_array = Skin_dummy_array[1:, :, :]

# Bone1 & 2
Bone1_dummy_array = np.zeros([1, int(Bone1_points/rand_sam_num), 3])
Bone2_dummy_array = np.zeros([1, int(Bone2_points/rand_sam_num), 3])

if rot_increasing_number == 0:
    all_inc_up_rand_Bone1_array = all_inc_up_rand_Bone1_array
    all_inc_up_rand_Bone2_array = all_inc_up_rand_Bone2_array
else:
    for i in range(all_inc_up_rand_Bone1_array.shape[0]):
        x_angle, y_angle, z_angle = Functions.make_rotate_angle(rot_increasing_number,
                                                                -10.0, 10.0,
                                                                -10.0, 10.0,
                                                                -10.0, 10.0)
        Bone1_temp = Functions_mk2.make_all_rotation_mk2(rot_increasing_number, x_angle, y_angle, z_angle,
                                                         all_inc_up_rand_Bone1_array[i])
        Bone2_temp = Functions_mk2.make_all_rotation_mk2(rot_increasing_number, x_angle, y_angle, z_angle,
                                                         all_inc_up_rand_Bone2_array[i])
        Bone1_dummy_array = np.concatenate((Bone1_dummy_array, Bone1_temp), axis=0)
        Bone2_dummy_array = np.concatenate((Bone2_dummy_array, Bone2_temp), axis=0)
    all_inc_up_rand_Bone1_array = Bone1_dummy_array[1:, :, :]
    all_inc_up_rand_Bone2_array = Bone2_dummy_array[1:, :, :]

# tar
tar_dummy_array = np.zeros([1, 1, 3])

if rot_increasing_number == 0:
    all_inc_rand_tar_arrays = all_inc_rand_tar_arrays
else:
    for i in range(all_inc_rand_tar_arrays.shape[0]):
        temp = np.tile(all_inc_rand_tar_arrays[i], [rot_increasing_number, 1, 1])
        tar_dummy_array = np.concatenate((tar_dummy_array, temp), axis=0)
    all_inc_rand_tar_arrays = np.delete(tar_dummy_array, 0, axis=0)

case = np.concatenate((all_inc_up_rand_Skin_array, all_inc_up_rand_Bone1_array), axis=1)
case = np.concatenate((case, all_inc_up_rand_Bone2_array), axis=1)
case = np.concatenate((case, all_inc_rand_tar_arrays), axis=1)

all_points = int((skin_points + Bone1_points + Bone2_points) / rand_sam_num)

all_case = np.zeros([1, all_points + 1, 3])

for i in range(case.shape[0]):
    copy_case = np.copy(case[i][:all_points, :])
    x_offset, y_offset, z_offset = Functions.make_offset_distance(tar_increasing_number - 1, tar_x[0], tar_x[1],
                                                                  tar_y[0], tar_y[1], tar_z[0], tar_z[1])
    all_Target = Functions.make_all_offset(case[i][all_points], x_offset, y_offset, z_offset, tar_increasing_number - 1)
    copy_lists = []
    for j in range(tar_increasing_number - 1):
        temp_case = np.concatenate((copy_case, np.expand_dims(all_Target[j], axis=0)), axis=0)
        copy_lists.append(temp_case)

    copy_arrays = np.array(copy_lists)
    rand_tar_case = np.concatenate((np.expand_dims(case[i], axis=0), copy_arrays), axis=0)
    all_case = np.concatenate((all_case, rand_tar_case), axis=0)
case = all_case[1:case.shape[0] * tar_increasing_number + 1, :, :]
print(case.shape)

# if rand_sam_num == 1:
#     case = np.concatenate((origin_case, case), axis=0)
print(case.shape)


# ############################## make data folder ######################################################################
max_data_num = len(os.walk('./data/').__next__()[1]) + 1
if not os.path.isdir(f'./data/data{max_data_num}'):  # 해당 경로 내에 폴더가 없으면,
    os.makedirs(f'./data/data{max_data_num}')  # 그 폴더를 만들기


# ############################## make file #############################################################################
np.save(f'./data/data{max_data_num}/data', case)
# np.save(f'./data/data{max_data_num}/number_of_data', case.shape[0])
np.save(f'./data/data{max_data_num}/num_bone1_points', Bone1_rand_sam_num)
np.save(f'./data/data{max_data_num}/num_bone2_points', Bone2_rand_sam_num)
np.save(f'./data/data{max_data_num}/num_skin_points', Skin_rand_sam_num)

file = open(f'./data/data{max_data_num}/information.txt', 'w', encoding='utf8')
line = f"set range : {first_set_num} ~ {last_set_num}, total {last_set_num - first_set_num + 1} sets\n\n" \
    f"number of Skin point : {int(skin_points/rand_sam_num)}\n" \
    f"size of Bone1 : {int(Bone1_points/rand_sam_num)}\n" \
    f"size of Bone2 : {int(Bone2_points/rand_sam_num)}\n\n" \
    f"< increasing process >\n" \
    f"(translate {trans_increasing_number}) X (rotation {rot_increasing_number}) X (up random {rand_sam_num})\n" \
    f"number of random target : {tar_increasing_number}\n\n" \
    f"number of data : {case.shape[0]}\n\n"
file.write(line)
print('\n\n추가 정보 기입')
string = str(input())
file.write(f'추가 정보: {string}')

file.close()

exit()