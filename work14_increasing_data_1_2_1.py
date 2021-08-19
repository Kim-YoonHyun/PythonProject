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
# _8_1
# class module 저장

if __name__ == '__main__':
    import numpy as np
    import os
    import functions_my
    import random
    import math
    import dill  # _7
    import pandas as pd  # _8
    import Augmentation  # _8
    import warnings

    warnings.filterwarnings(action='ignore')

    # basic parameter
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

    # set 번호 범위 지정
    stl_set_order_range = list(range(first_stl_set_num, last_stl_set_num + 1))
    stl_set_path = f'stl/'

    # 각 데이터별 class 세팅
    target_data = Augmentation.DataInformation('target')
    bone1_data = Augmentation.DataInformation('bone1')
    bone2_data = Augmentation.DataInformation('bone2')
    skin_data = Augmentation.DataInformation('skin')

    # <초기 데이터 불러오기> ------------------------------------------------------------
    for stl_set_num in range(first_stl_set_num, last_stl_set_num + 1):
        file_list = os.listdir(f'{stl_set_path}/set{stl_set_num}')
        bone_flag = 1
        for file_name in file_list:
            if '.stl' in file_name:
                stl_name = file_name.split('.')[-2]
                stl_vertices_list, num_of_stl_points = functions_my.read_stl(f'{stl_set_path}/set{stl_set_num}',
                                                                             stl_name)
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
    target_data.data_vertices_list = [
        np.tile(np.array(target_data.data_vertices_list)[i], [num_of_rand_sam, 1, 1])[j].tolist()
        for i in range(len(target_data.data_vertices_list))
        for j in range(num_of_rand_sam)
    ]
    target_data.num_of_data *= num_of_rand_sam
    target_data.rand_sam_status = f'copyX{num_of_rand_sam}'
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

    skin_array = np.array(skin_data.data_vertices_list)
    np.save(f'./skin_array', skin_array)
    exit()

    # < rotation > ------------------------------------------------
    if num_of_rot > 1:
        print('rotation...')

        # rotation matrix
        all_rot_matrices = []
        for j in range(target_data.num_of_data):
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

        # rotation
        target_data.rotation(num_of_rot, all_rot_matrices)
        bone1_data.rotation(num_of_rot, all_rot_matrices)
        bone2_data.rotation(num_of_rot, all_rot_matrices)
        skin_data.rotation(num_of_rot, all_rot_matrices)
    else:
        print('not rotation')

    # <data 정리>------------------------------------------------------------
    all_data = [target_data, bone1_data, bone2_data, skin_data]
    num_of_data_list, num_of_data_point_list, array_size_list, data_name_list = [], [], [], []
    up_sam_status_list, rand_sam_status_list, trans_status_list, rot_status_list = [], [], [], []
    all_data_vertices_list = []

    for data in all_data:
        num_of_data_list.append(data.num_of_data)
        num_of_data_point_list.append(data.num_of_data_points)
        array_size_list.append(np.array(data.data_vertices_list).shape)
        data_name_list.append(data.data_name)
        up_sam_status_list.append(data.up_sam_status)
        rand_sam_status_list.append(data.rand_sam_status)
        trans_status_list.append(data.trans_status)
        rot_status_list.append(data.rot_status)
        all_data_vertices_list.append(np.array(data.data_vertices_list))

    df = pd.DataFrame({
        '데이터 갯수': num_of_data_list, '포인트 갯수': num_of_data_point_list, 'shape': array_size_list,
        'up sam 상태': up_sam_status_list, 'rand sam 상태': rand_sam_status_list, 'trans 상태': trans_status_list,
        'rot 상태': rot_status_list
    }, index=data_name_list)
    df.index.name = '이름'
    print(df)

    input_data = np.concatenate(all_data_vertices_list, axis=1)
    print(f'input data size: {input_data.shape}')

    # case = np.concatenate((all_inc_up_rand_Skin_array, all_inc_up_rand_Bone1_array), axis=1)
    # case = np.concatenate((case, all_inc_up_rand_Bone2_array), axis=1)
    # case = np.concatenate((case, all_inc_rand_tar_arrays), axis=1)
    #
    # all_points = int((skin_points + Bone1_points + Bone2_points) / rand_sam_num)
    #
    # all_case = np.zeros([1, all_points + 1, 3])
    #
    # for i in range(case.shape[0]):
    #     copy_case = np.copy(case[i][:all_points, :])
    #     x_offset, y_offset, z_offset = Functions.make_offset_distance(tar_increasing_number - 1, tar_x[0], tar_x[1],
    #                                                                   tar_y[0], tar_y[1], tar_z[0], tar_z[1])
    #     all_Target = Functions.make_all_offset(case[i][all_points], x_offset, y_offset, z_offset, tar_increasing_number - 1)
    #     copy_lists = []
    #     for j in range(tar_increasing_number - 1):
    #         temp_case = np.concatenate((copy_case, np.expand_dims(all_Target[j], axis=0)), axis=0)
    #         copy_lists.append(temp_case)
    #
    #     copy_arrays = np.array(copy_lists)
    #     rand_tar_case = np.concatenate((np.expand_dims(case[i], axis=0), copy_arrays), axis=0)
    #     all_case = np.concatenate((all_case, rand_tar_case), axis=0)
    # case = all_case[1:case.shape[0] * tar_increasing_number + 1, :, :]
    # print(case.shape)
    #
    # # if rand_sam_num == 1:
    # #     case = np.concatenate((origin_case, case), axis=0)
    # print(case.shape)
    #

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
    df.to_csv(f'{data_save_path}/data_information.csv', encoding='utf-8-sig')
    
    # npy 저장
    np.save(f'{data_save_path}/input_data', input_data)
    
    # txt 저장
    with open(f'{data_save_path}/stl set range = {first_stl_set_num} ~ {last_stl_set_num}.txt', 'w',
              encoding='utf8') as file:
        file.write(f'이 데이터는 stl set{first_stl_set_num} 부터 {last_stl_set_num} 까지,\n'
                   f'총 {last_stl_set_num - first_stl_set_num + 1}개 로 만들어진 data 입니다.')
    exit()
    # file = open(f'./data/data{max_data_num}/information.txt', 'w', encoding='utf8')
    # line = f"set range : {first_set_num} ~ {last_set_num}, total {last_set_num - first_set_num + 1} sets\n\n" \
    #     f"number of Skin point : {int(skin_points/rand_sam_num)}\n" \
    #     f"size of Bone1 : {int(Bone1_points/rand_sam_num)}\n" \
    #     f"size of Bone2 : {int(Bone2_points/rand_sam_num)}\n\n" \
    #     f"< increasing process >\n" \
    #     f"(translate {trans_increasing_number}) X (rotation {rot_increasing_number}) X (up random {rand_sam_num})\n" \
    #     f"number of random target : {tar_increasing_number}\n\n" \
    #     f"number of data : {case.shape[0]}\n\n"
    # file.write(line)
    # print('\n\n추가 정보 기입')
    # string = str(input())
    # file.write(f'추가 정보: {string}')
    # file.close()