# 기존의 데이터에서 폴더명을 label로 변경
# _4
# score 를 Min 에서 Max 로 변경

# _5
# 저장되는 데이터 변경

# _6
# word 설명과 맞추기, 반복 편의성 수정

import numpy as np
import os


# ############################## range setting #########################################################################
# model&data number
max_data_num = len(os.walk('./data/').__next__()[1])


# ############################## Enter the parameter ###################################################################
while True:
    # data 번호 입력
    print(f'\n\n\nA. Enter the data number for labeling (1 ~ {max_data_num})')
    try:
        data_num = int(input())
        if data_num > max_data_num or data_num < 1:
            continue
        else:
            data_num = data_num
    except ValueError:
        continue

    # labeling 유무 확인 및 path 설정
    path = f'./data/data{data_num}'
    labeled_data_path = f'{path}(Label)'
    if not(os.path.isdir(labeled_data_path)):
        os.renames(path, labeled_data_path)
    else:
        print('this data is already labeled.')
        continue


# ############################## load data #############################################################################
    data = np.load(labeled_data_path + f'/data.npy')
    num_B1_points = np.load(labeled_data_path + f'/num_bone1_points.npy')
    num_B2_points = np.load(labeled_data_path + f'/num_bone2_points.npy')
    num_sur_points = np.load(labeled_data_path + f'/num_skin_points.npy')
    # num_tar_points, array_tar_set1 = 1, np.array([[-7., 5., 0.]])  # 현재로선 이 값은 변경 불가
    # _,              array_tar_set2 = 1, np.array([[ 0., 0., 0.]])  # 현재로선 이 값은 변경 불가
    num_tar_points = 1
    print('old data : ', data.shape)


# ############################## labeling ##############################################################################
    # 각 모델에 대한 label 값
    sur_label = 1
    B1_label = 2
    B2_label = 3
    tar_label = 4

    # 각 모델 point 갯수만큼 label 을 복사
    sur_label_array = np.full((data.shape[0], num_sur_points, 1), sur_label)
    B1_label_array = np.full((data.shape[0], num_B1_points, 1), B1_label)
    B2_label_array = np.full((data.shape[0], num_B2_points, 1), B2_label)
    tar_label_array = np.full((data.shape[0], num_tar_points, 1), tar_label)

    # labeling
    temp = np.concatenate((sur_label_array, B1_label_array), axis=1)
    temp2 = np.concatenate((temp, B2_label_array), axis=1)
    all_label_array = np.concatenate((temp2, tar_label_array), axis=1)
    labeled_data = np.concatenate((data, all_label_array), axis=-1)

    print('label data : ', labeled_data.shape)


    # ############################## make file #############################################################################
    np.save(labeled_data_path + f'/data', labeled_data)
    file = open(labeled_data_path + f'/information.txt', 'a', encoding='utf8')
    line = f'\n\n\nsurface label : {sur_label}\n' \
        f'Bone1 label : {B1_label}\n' \
        f'Bone2 label : {B2_label}\n' \
        f'target label : {tar_label}\n\n'
    file.write(line)
    print('\n\n추가 정보 기입')
    string = str(input())
    file.write(f'추가 정보: {string}')

    file.close()