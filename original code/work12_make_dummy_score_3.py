# Up random sampling 을 통해 만든 data 의 score 계산이 되질 않는다.
# labeling 구분 여부 추가

# _3
# word 설명과 맞추기, 반복 편의성 수정

import numpy as np
import os


# ############################## range setting #########################################################################
# model&data number
max_data_num = len(os.walk('./data/').__next__()[1])


# ############################## Enter the parameter ###################################################################
while True:
    # data 번호 입력
    print(f'\n\n\nA. Enter the data number for make dummy (1 ~ {max_data_num})')
    try:
        data_num = int(input())
        if data_num > max_data_num or data_num < 1:
            continue
        else:
            data_num = data_num
    except ValueError:
        continue

    print(f'B. use labeling? (on or anything)')
    use_label_data = str(input())


# ############################## path 설정 #############################################################################
    if use_label_data == 'on':
        path = f'./data/data{data_num}(Label)'
    else:
        path = f'./data/data{data_num}'


# ############################## Read data #############################################################################
    data = np.load(f'{path}/data.npy')
    score = np.load(f'{path}/score.npy')
    num_sur_points = np.load(f'{path}/num_skin_points.npy')
    num_B1_points = np.load(f'{path}/num_bone1_points.npy')
    num_B2_points = np.load(f'{path}/num_bone2_points.npy')
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
    np.save(path + f'/score_add_dummy', score)
    print(score.shape)