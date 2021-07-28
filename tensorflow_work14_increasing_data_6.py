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

# _5 >>> 20210618 지금까지의 모든 결과를 얻어낸 코드
# 정리 word 파일과 맞추기
# Functions import 방식 변경

# _6
# sys 삭제
# 데이터 폴더 번호 자동생성 제거, 데이터 번호 설정 변수 추가

import numpy as np
import os
import Functions
import Functions_mk2


# ############################## parameter #############################################################################
# 데이터 번호 설정 변수
data_number = 34
if os.path.isdir(f'./data/data{data_number}'):  # 해당 경로 내에 폴더가 있으면,
    print('이미 해당 번호의 데이터가 존재합니다.')  # 실행 종료
    exit()
else:
    os.makedirs(f'./data/data{data_number}')  # 그 폴더를 만들기

# A. stl 파일 불러오기. 현재 사용 중인 set 은 32번 부터이다.
first_set_num = 32  # 불러올 set 의 시작 번호
last_set_num = 33  # 불러올 set 의 마지막 번호

# B. Augmentation: translate
trans_increasing_number = 2  # translate 를 통해 증가시킬 배수
trans_x = [-3.0, 3.0]  # x 축에 대한 최대 translate 범위
trans_y = [-3.0, 3.0]  # y 축에 대한 최대 translate 범위
trans_z = [-3.0, 3.0]  # z 축에 대한 최대 translate 범위

# B. Augmentation: rotation
rot_increasing_number = 2  # rotation 을 통해 증가시킬 배수t
rot_x = [-5.0, 5.0]  # x 축에 대한 최대 rotation 범위
rot_y = [-5.0, 5.0]  # y 축에 대한 최대 rotation 범위
rot_z = [-5.0, 5.0]  # z 축에 대한 최대 rotation 범위

# C. up random sampling
skin_points = 3000  # 증가시킬 skin 의 point 갯수
Bone1_points = 1500  # 증가시킬 Bone1 의 point 갯수
Bone2_points = 1500  # 증가시킬 Bone2 의 point 갯수
max_length = 0.1  # 새로운 점을 생성하기 위한 기준 길이.
rand_sam_num = 2

# D. 랜덤 타겟 생성
tar_x = [-18.0, 18.0]
tar_y = [-3.0, 7.0]
tar_z = [-2.0, 2.0]
tar_increasing_number = 2


# ############################## Make dict #############################################################################
# property 넘버링
# property0 : Disc
# property1 : Bone1
# property2 : Bone2
# property3 : Skin
target_property_num = 4

# =======================================================
# set 번호 범위 지정
set_num = []
for i in range(first_set_num, last_set_num + 1):
    set_num.append(i)
set_num = np.array(set_num)

# 지정한 범위의 stl 전부 불러오기
# set, _ = Functions.Make_set(set_num, target_property_num)
set, a = Functions_mk2.make_set_mk2(set_num, target_property_num, './stl/set', '.stl')

# 모든 set 의 model 정보 print
for i in range(set_num.shape[0]):
    print(f"size of {set[set_num[i]]['property0']['name of model']} : {set[set_num[i]]['property0']['array'].shape}")
    print(f"size of {set[set_num[i]]['property1']['name of model']} : {set[set_num[i]]['property1']['array'].shape}")
    print(f"size of {set[set_num[i]]['property2']['name of model']} : {set[set_num[i]]['property2']['array'].shape}")
    print(f"size of {set[set_num[i]]['property3']['name of model']} : {set[set_num[i]]['property3']['array'].shape}")
    print(f"number of all points = {set[set_num[i]]['property0']['number of point'] + set[set_num[i]]['property1']['number of point'] + set[set_num[i]]['property2']['number of point'] + set[set_num[i]]['property3']['number of point']}")
    print('\n')


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


# ############################## save data #############################################################################
np.save(f'./data/data{data_number}/data', case)
np.save(f'./data/data{data_number}/num_bone1_points', Bone1_rand_sam_num)
np.save(f'./data/data{data_number}/num_bone2_points', Bone2_rand_sam_num)
np.save(f'./data/data{data_number}/num_skin_points', Skin_rand_sam_num)

file = open(f'./data/data{data_number}/information.txt', 'w', encoding='utf8')
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
