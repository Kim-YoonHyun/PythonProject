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


import numpy as np
import os
import my_functions
import random
# import Functions
# import Functions_mk2

import vtk

# ############################## parameter #############################################################################
# A. stl 파일 불러오기. 현재 사용 중인 set 은 32번 부터이다.
first_stl_set_num = 32  # 불러올 set 의 시작 번호
last_stl_set_num = 35  # 불러올 set 의 마지막 번호

# B. Augmentation: translate
num_of_trans_increasing = 2  # translate 를 통해 증가시킬 배수
trans_x = [-3.0, 3.0]  # x 축에 대한 최대 translate 범위
trans_y = [-3.0, 3.0]  # y 축에 대한 최대 translate 범위
trans_z = [-3.0, 3.0]  # z 축에 대한 최대 translate 범위

# B. Augmentation: rotation
num_of_rot_increasing = 2  # rotation 을 통해 증가시킬 배수
rot_x = [-5.0, 5.0]  # x 축에 대한 최대 rotation 범위
rot_y = [-5.0, 5.0]  # y 축에 대한 최대 rotation 범위
rot_z = [-5.0, 5.0]  # z 축에 대한 최대 rotation 범위

# C. up random sampling
num_of_skin_points = 3000  # 증가시킬 skin 의 point 갯수
num_of_Bone1_points = 1500  # 증가시킬 Bone1 의 point 갯수
num_of_Bone2_points = 1500  # 증가시킬 Bone2 의 point 갯수
max_length = 0.1  # 새로운 점을 생성하기 위한 기준 길이.
num_of_rand_sam = 2

# D. 랜덤 타겟 생성
tar_x = [-18.0, 18.0]
tar_y = [-3.0, 7.0]
tar_z = [-2.0, 2.0]
num_of_tar_increasing = 2


# ############################## Make dict #############################################################################
# property 넘버링
# property0 : Disc
# property1 : Bone1
# property2 : Bone2
# property3 : Skin
target_property_num = 4


# =======================================================
# set 번호 범위 지정
stl_set_order_range = list(range(first_stl_set_num, last_stl_set_num + 1))
stl_set_path = f'stl/set'

# stl_set_property_list = []
# for stl_set_num in range(first_stl_set_num, last_stl_set_num + 1):
#     # 빈 리스트 준비
#     stl_name_list = []
#     num_of_stl_points_list = []
#     stl_vertices_list = []
#
#     # stl set 번호 별 이름, 좌표, 포인트 갯수 불러오기
#     file_list = os.listdir(f'{stl_set_path}{stl_set_num}')
#     for i in range(len(file_list)):
#         if file_list[i].split('.')[-1] == 'stl':
#             stl_name = file_list[i].split('.')[-2]
#             stl_vertices, num_of_stl_points = my_functions.read_stl(f'{stl_set_path}{stl_set_num}', stl_name)
#             stl_name_list.append(stl_name)
#             num_of_stl_points_list.append(num_of_stl_points)
#             stl_vertices_list.append(stl_vertices)
#
#     # key list를 통해 불러온 정보와 겹쳐서 dict 만들기
#     key_list = ['stl_name', 'num_of_stl_points', 'stl_vertices']
#     new_list = list(
#         [stl_name_list.pop(0), num_of_stl_points_list.pop(0), stl_vertices_list.pop(0)]
#         for i in range(len(stl_name_list))
#     )
#     property = list(dict(zip(key_list, new_list[i])) for i in range(len(new_list)))
#
#     # 모든 stl set의 property dictionary 합치기
#     stl_set_property_list.append(property)
#
# for i in range(last_stl_set_num - first_stl_set_num + 1):
#     for j in range(4):
#             print(f"{stl_set_property_list[i][j]['stl_name'].rjust(10)}", end=', ')
#             print(f"{stl_set_property_list[i][j]['num_of_stl_points']:4d}", end=', ')
#             print(f"{np.array(stl_set_property_list[i][j]['stl_vertices']).shape}")
#     print()
#
# exit()

# stl_set32_property = [
#     {'stl name':'D1_set32', 'stl point num':409, 'stl vertices':[[1, 2, 3]]},
#     {'stl name':'L3_set32', 'stl point num':1040, 'stl vertices':[[5, 3, 2]]},
#     {'stl name':'L4_set32', 'stl point num':1011, 'stl vertices':[[5, 3, 2]]}
# ]
#
# stl_set33_property = [
#     {'stl name':'D1_set33', 'stl point num':209, 'stl vertices':[[1, 2, 3]]},
#     {'stl name':'L3_set33', 'stl point num':940, 'stl vertices':[[5, 3, 2]]},
#     {'stl name':'L4_set33', 'stl point num':911, 'stl vertices':[[5, 3, 2]]}
# ]

stl_set_property_list = []

for stl_set_num in range(first_stl_set_num, last_stl_set_num + 1):
    # 빈 리스트 준비
    # num_of_stl_points_list = []
    # all_stl_vertices_list = []

    # stl set 번호 별 이름, 좌표, 포인트 갯수 불러오기
    file_list = os.listdir(f'{stl_set_path}{stl_set_num}')
    for i in range(len(file_list)):
        if file_list[i].split('.')[-1] == 'stl':
            stl_vertices_list, num_of_stl_points = my_functions.read_stl(f'{stl_set_path}{stl_set_num}', file_list[i].split('.')[-2])
            stl_vertices_array = np.array(stl_vertices_list)
            index_array = np.array([j for j in range(num_of_stl_points)])
            # Disc 데이터를 target 데이터로 바꾸기
            if file_list[i].split('.')[-2].split('_')[0][0] == 'D':
                stl_vertices_array_center = np.average(stl_vertices_array, axis=0)
                stl_vertices_array_min = np.min(stl_vertices_array, axis=0)
                while  stl_vertices_array_center[1] > stl_vertices_array_min[1] - 1:
                    stl_vertices_array_center[1] -= 1
                stl_vertices_array_center = np.expand_dims(stl_vertices_array_center, axis=0)

            # 그외 skin, bone 데이터의 경우 >>> 작업중
            else:
                def euclidean_distance(start, end):
                    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
                    return value

                for i in range(num_of_stl_points):
                    # rand_index = random.randint(0, num_of_stl_points)
                    rand_index = 2
                    rand_vertex = stl_vertices_array[rand_index]

                    stl_vertices_array = np.delete(stl_vertices_array, rand_index, axis=0)
                    index_array = np.delete(index_array, rand_index, axis=0)
                    euclidean_distance = euclidean_distance(rand_vertex, stl_vertices_array)
                    print(np.min(euclidean_distance))
                    c = dict(zip(euclidean_distance, index_array))
                    print(c[np.min(euclidean_distance)])
                    print(c)
                    # print(sorted(c.values()))
                    # print(c)
                    # print(c[0])
                    exit()




                    # euclidean_distance_sorted = sorted(euclidean_distance)
                #     print(euclidean_distance_sorted)

                # a = [1, 2, 3]
                # b = np.array([
                #     [2, -1, 4],
                #     [6,  2, 0]
                #      ])
                # print(euclidean_distance(rand_vertex, stl_vertices_array).shape)
                # print(np.subtract(a, b))
                # print(np.square(np.subtract(a, b)))
                # print(np.sum(np.square(np.subtract(a, b)), axis=-1))
                # print(np.sqrt(np.sum(np.square(np.subtract(a, b)), axis=-1)))

                exit()









            # num_of_stl_points_list.append(num_of_stl_points)
            # all_stl_vertices_list.append(stl_vertices_list)

    # key list를 통해 불러온 정보와 겹쳐서 dict 만들기
    key_list = ['num_of_stl_points', 'stl_vertices']
    new_list = list(
        [num_of_stl_points_list.pop(0), all_stl_vertices_list.pop(0)]
        for i in range(len(num_of_stl_points_list))
    )
    property = list(dict(zip(key_list, new_list[i])) for i in range(len(new_list)))

    # 모든 stl set의 property dictionary 합치기
    stl_set_property_list.append(property)

for i in range(last_stl_set_num - first_stl_set_num + 1):
    for j in range(4):
            print(f"{stl_set_property_list[i][j]['num_of_stl_points']:4d}", end=', ')
            print(f"{np.array(stl_set_property_list[i][j]['stl_vertices']).shape}")
    print()

exit()

# # 모든 set 의 model 정보 print
# for i in range(set_num.shape[0]):
#     print(f"size of {set[set_num[i]]['property0']['name of model']} : {set[set_num[i]]['property0']['array'].shape}")
#     print(f"size of {set[set_num[i]]['property1']['name of model']} : {set[set_num[i]]['property1']['array'].shape}")
#     print(f"size of {set[set_num[i]]['property2']['name of model']} : {set[set_num[i]]['property2']['array'].shape}")
#     print(f"size of {set[set_num[i]]['property3']['name of model']} : {set[set_num[i]]['property3']['array'].shape}")
#     print(f"number of all points = {set[set_num[i]]['property0']['number of point'] + set[set_num[i]]['property1']['number of point'] + set[set_num[i]]['property2']['number of point'] + set[set_num[i]]['property3']['number of point']}")
#     print('\n')

# ############################## Increasing data #######################################################################
# 모든 set 의 skin, Bone1, Bone2 를 따로 모음. 모델마다 point 갯수가 다르기 때문.
all_origin_Skin_array = []
all_origin_Bone1_array = []
all_origin_Bone2_array = []
all_origin_tar_array = []
for i in range(set_num.shape[0]):
    all_origin_Bone1_array.append(set[set_num[i]]['property1']['array'])
    all_origin_Bone2_array.append(set[set_num[i]]['property2']['array'])
    all_origin_Skin_array.append(set[set_num[i]]['property3']['array'])
    all_origin_tar_array.append(set[set_num[i]]['property4']['array'])
all_origin_Bone1_array = np.array(all_origin_Bone1_array)
all_origin_Bone2_array = np.array(all_origin_Bone2_array)
all_origin_Skin_array = np.array(all_origin_Skin_array)
all_origin_tar_array = np.array(all_origin_tar_array)

# =======================================================
# translate & Up sampling
Bone1_dummy_array = np.zeros([1, Bone1_points, 3])
Bone2_dummy_array = np.zeros([1, Bone2_points, 3])
Skin_dummy_array = np.zeros([1, skin_points, 3])
tar_dummy_array = np.zeros([1, 1, 3])

# original data processing
all_origin_up_Skin_array = Functions.up_sampling(all_origin_Skin_array, skin_points, max_length)
all_origin_up_Bone1_array = Functions.up_sampling(all_origin_Bone1_array, Bone1_points, max_length)
all_origin_up_Bone2_array = Functions.up_sampling(all_origin_Bone2_array, Bone2_points, max_length)

origin_case = np.concatenate((all_origin_up_Skin_array, all_origin_up_Bone1_array), axis=1)
origin_case = np.concatenate((origin_case, all_origin_up_Bone2_array), axis=1)
origin_case = np.concatenate((origin_case, all_origin_tar_array), axis=1)


print(all_origin_up_Skin_array.shape)
print(all_origin_up_Bone1_array.shape)
print(all_origin_up_Bone2_array.shape)
print(origin_case.shape)

# translate
if trans_increasing_number == 0:
    all_inc_up_Skin_array = all_origin_up_Skin_array
    all_inc_up_Bone1_array = all_origin_up_Bone1_array
    all_inc_up_Bone2_array = all_origin_up_Bone2_array
    all_inc_tar_array = all_origin_tar_array
else:
    for i in range(len(set)):
        x_offset, y_offset, z_offset = Functions.make_offset_distance(trans_increasing_number, trans_x[0], trans_x[1],
                                                                      trans_y[0], trans_y[1], trans_z[0], trans_z[1])

        # Skin
        Skin_temp = Functions.make_all_offset(all_origin_up_Skin_array[i], x_offset, y_offset, z_offset,
                                              trans_increasing_number)
        Skin_dummy_array = np.concatenate((Skin_dummy_array, Skin_temp), axis=0)

        # # Bone1
        Bone1_temp = Functions.make_all_offset(all_origin_up_Bone1_array[i], x_offset, y_offset, z_offset,
                                               trans_increasing_number)
        Bone1_dummy_array = np.concatenate((Bone1_dummy_array, Bone1_temp), axis=0)

        # Bone2
        Bone2_temp = Functions.make_all_offset(all_origin_up_Bone2_array[i], x_offset, y_offset, z_offset,
                                               trans_increasing_number)
        Bone2_dummy_array = np.concatenate((Bone2_dummy_array, Bone2_temp), axis=0)

        # tar
        tar_temp = Functions.make_all_offset(all_origin_tar_array[i], x_offset, y_offset, z_offset,
                                             trans_increasing_number)
        tar_dummy_array = np.concatenate((tar_dummy_array, tar_temp), axis=0)

        # dummy array 제거
        all_inc_up_Skin_array = Skin_dummy_array[1:, :, :]
        all_inc_up_Bone1_array = Bone1_dummy_array[1:, :, :]
        all_inc_up_Bone2_array = Bone2_dummy_array[1:, :, :]
        all_inc_tar_array = tar_dummy_array[1:, :, :]

# =======================================================
# random sampling
Skin_rand_sam_num = int(skin_points/rand_sam_num)
Bone1_rand_sam_num = int(Bone1_points/rand_sam_num)
Bone2_rand_sam_num = int(Bone2_points/rand_sam_num)
Skin_dummy_arrays = np.zeros([1, Skin_rand_sam_num, 3])
Bone1_dummy_arrays = np.zeros([1, Bone1_rand_sam_num, 3])
Bone2_dummy_arrays = np.zeros([1, Bone2_rand_sam_num, 3])


all_inc_up_rand_Skin_array, _ = Functions.random_sampling(all_inc_up_Skin_array, Skin_dummy_arrays, Skin_rand_sam_num)
all_inc_up_rand_Bone1_array, Bone1_tile = Functions.random_sampling(all_inc_up_Bone1_array, Bone1_dummy_arrays,
                                                                    Bone1_rand_sam_num)
all_inc_up_rand_Bone2_array, _ = Functions.random_sampling(all_inc_up_Bone2_array, Bone2_dummy_arrays,
                                                           Bone2_rand_sam_num)
all_inc_rand_tar_arrays = np.zeros([1, 1, 3])

for i in range(all_inc_tar_array.shape[0]):
    temp = np.tile(all_inc_tar_array[i], [Bone1_tile[i], 1, 1])
    all_inc_rand_tar_arrays = np.concatenate((all_inc_rand_tar_arrays, temp), axis=0)
all_inc_rand_tar_arrays = np.delete(all_inc_rand_tar_arrays, 0, axis=0)

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
