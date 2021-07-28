# 기존의 데이터에서 폴더명을 label로 변경
# _4
# score 를 Min 에서 Max 로 변경

# _5
# 저장되는 데이터 변경

# _6 >>> 20210618 지금까지의 모든 결과를 추출한 코드
# word 설명과 맞추기, 반복 편의성 수정

# _7
# 데이터를 선택할때 폴더 최대갯수가 아닌 폴더 이름을 통해 범위를 설정
# 폴더 이름 변경 삭제, 라벨 데이터 npy 파일 추가 저장

import numpy as np
import os


# ############################## range setting #########################################################################
# folder 번호 불러오기
folder_name_list = os.listdir('./data')
folder_number_list = []
for i in range(len(folder_name_list)):
    folder_number_list.append(int(folder_name_list[i][4:]))
folder_number_list.sort()

# ############################## Enter the parameter ###################################################################
while True:
    print('data number list')
    print(folder_number_list)
    
    # 리스트에 없는 번호 선택시 오류 처리
    try:
        number_index = folder_number_list.index(int(input('Enter the number : ')))
    except ValueError:
        print('\n\n\n리스트에 없는 번호입니다.')
        continue
    data_number = folder_number_list[number_index]
    
    # data path 설정
    data_path = './data/data' + str(data_number)

    # 이미 Label 된 data 인 경우
    if os.path.isfile(data_path + '/data_labeled.npy'):
        print('\n\n\n이미 label 된 data 입니다.')
        continue


# ############################## load data #############################################################################
    data = np.load(data_path + f'/data.npy')
    num_B1_points = np.load(data_path + f'/num_bone1_points.npy')
    num_B2_points = np.load(data_path + f'/num_bone2_points.npy')
    num_sur_points = np.load(data_path + f'/num_skin_points.npy')
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
    np.save(data_path + f'/data_labeled', labeled_data)
    file = open(data_path + f'/information.txt', 'a', encoding='utf8')
    line = f'\n\n\nsurface label : {sur_label}\n' \
        f'Bone1 label : {B1_label}\n' \
        f'Bone2 label : {B2_label}\n' \
        f'target label : {tar_label}\n\n'
    file.write(line)
    print('\n\n추가 정보 기입')
    string = str(input())
    file.write(f'추가 정보: {string}')

    file.close()