# Up random sampling 을 통해 만든 data 의 score 계산이 되질 않는다.
# labeling 구분 여부 추가

# _3 >>> 20210618 지금까지 모든 결과를 추출한 코드
# word 설명과 맞추기, 반복 편의성 수정

# _4
# 데이터를 선택할때 폴더 최대갯수가 아닌 폴더 이름을 통해 범위를 설정
# 데이터는 기본적으로 label 된 데이터만 불러오도록 변경

import numpy as np
import os


# ############################## range setting #########################################################################
folder_name_list = os.listdir('./data')
folder_number_list = []
for i in range(len(folder_name_list)):
    folder_number_list.append(int(folder_name_list[i][4:]))
folder_number_list.sort()


# ############################## Enter the parameter ###################################################################
while True:
    # data 번호 입력
    print('data number for make dummy data')
    print(folder_number_list)

    # 입력 된 번호가 리스트에 없을 경우 오류 처리
    try:
        index = folder_number_list.index(int(input('Enter the data number for make dummy : ')))
        data_number = folder_number_list[index]
    except ValueError:
        print('해당 번호는 리스트에 없습니다.')
        continue
    
    # data path 설정
    data_path = './data/data' + str(data_number)

    # 이미 dummy 가 만들어진 data일 경우 처리
    if os.path.isfile(data_path + '/score_add_dummy.npy'):
        print('\n\n\n이미 dummy가 계산된 data입니다.')
        continue
        
# ############################## Read data #############################################################################
    data = np.load(data_path + '/data_labeled.npy')    
    score = np.load(f'{data_path}/score.npy')
    num_sur_points = np.load(f'{data_path}/num_skin_points.npy')
    num_B1_points = np.load(f'{data_path}/num_bone1_points.npy')
    num_B2_points = np.load(f'{data_path}/num_bone2_points.npy')
    num_tar_points = 1
    num_all_data = data.shape[0]


# ############################## Make dummy data #######################################################################
    list = []
    for i in range(num_all_data):

        # Bone1 에 할당할 dummy score
        B1 = data[i][num_sur_points:num_sur_points + num_B1_points, 0:3]
        # 평균값을 통해 dummy score 계산
        B1 = np.average(B1, axis=-1)

        # Bone2 에 할당할 dummy score
        B2 = data[i][num_sur_points + num_B1_points:num_sur_points + num_B1_points + num_B2_points, 0:3]
        B2 = np.average(B2, axis=-1)

        # target 에 할당할 dummy score
        tar = data[i][num_sur_points + num_B1_points + num_B2_points:num_sur_points + num_B1_points + num_B2_points + num_tar_points, 0:3]
        tar = np.average(tar, axis=-1)

        # concatenate
        temp = np.concatenate((score[i], B1), axis=0)
        temp2 = np.concatenate((temp, B2), axis=0)
        temp3 = np.concatenate((temp2, tar), axis=0)
        list.append(temp3)

    score = np.array(list)
    np.save(data_path + f'/score_add_dummy', score)
    print(score.shape)