# _2
# 복잡한 계산 간결화

# _3
# score 최적점을 0 에서 1 로 변경
# _1
# epsilon 을 L1, L2, L3 각각의 최댓값으로 설정함

# _4 >>> 20210618 지금까지 모든 결과를 추출한 코드
# word 설명과 맞추기
# _4_1
# 실행전후 파라미터 변경, 편의성 수정
# Functions import 방식 변경

# _5
# sys 삭제
# 데이터 범위를 데이터 이름을 통해 추출

import numpy as np
import time
import os
import Functions


# ############################## parameter #############################################################################
while True:
    # 폴더리스트에서 폴더 번호 불러오기
    folder_name_list = os.listdir('./data')
    folder_number_list = []
    for i in range(len(folder_name_list)):
        folder_number_list.append(int(folder_name_list[i][4:]))
    folder_number_list.sort()

    # ----- Range setting
    print("Range of data number")
    print(folder_number_list)

    # 리스트에 입력한 번호가 없을 경우 오류 처리
    try:
        index = folder_number_list.index(int(input('A. Enter the data number for calculate score :')))
        data_number = folder_number_list[index]
    except ValueError:
        print('\n\n\n해당 번호는 리스트에 없습니다.')
        continue
    
    # data path 설정
    path = f'./data/data{data_number}'

    # 이미 score 가 계산된 경우 처리
    if os.path.isfile(path + '/score.npy'):
        print('\n\n\n이미 score 가 계산된 데이터입니다.')
        continue
    
    # score function 타입 설정
    switch = input('B. enter the score function type ( Max or Min ) :')
    if switch == 'Min':
        e = int(input('C. enter the epsilon value :'))

    start = time.time()

# #################################################### calculate #######################################################
    # data 불러오기
    data = np.load(f'{path}/data.npy')
    num_skin_points = np.load(f'{path}/num_skin_points.npy')
    num_B1_points = np.load(f'{path}/num_bone1_points.npy')
    num_B2_points = np.load(f'{path}/num_bone2_points.npy')
    num_tar_points = 1

    print('\nnumber of data:', data.shape[0])
    print('number of Skin points:', num_skin_points)
    print('number of Bone1 points:', num_B1_points)
    print('number of Bone2 points:', num_B2_points)

    num_all_data = data.shape[0]

    # =======================================================
    # surface 와 target 간의 모든 거리 구하기(L2)
    skin = data[:num_all_data, 0:num_skin_points, :]
    Bone1 = data[:num_all_data, num_skin_points:num_skin_points+num_B1_points, :]
    Bone2 = data[:num_all_data, num_skin_points+num_B1_points:num_skin_points+num_B1_points+num_B2_points, :]
    tar = data[:num_all_data, num_skin_points+num_B1_points+num_B2_points:num_skin_points+num_B1_points+num_B2_points+num_tar_points, :]
    L2 = Functions.euclidean_distance(start=skin, end=tar)

    # =======================================================
    # Bone1 및 Bone2 와 L2 간의 모든 거리 중 최소값(L2와 가장 가까운 Bone 좌표)구하기(L1, L3)
    L1 = []
    L3 = []
    for idx in range(num_all_data):
        print(f'{idx + 1}/{num_all_data}')
        L1.append(Functions.calculate_point_to_line_length(tar[idx], Bone1[idx], skin[idx]))
        L3.append(Functions.calculate_point_to_line_length(tar[idx], Bone2[idx], skin[idx]))
    L1 = np.array(L1)
    L3 = np.array(L3)

    new_L2 = []
    new_L1 = []
    new_L3 = []

    for i in range(num_all_data):
        if switch == 'Max':
            new_L2.append(np.add(L2[i], max(L2[i])))
            new_L1.append(np.add(L1[i], max(L1[i])))
            new_L3.append(np.add(L3[i], max(L3[i])))
            e2 = 'L2 max'
            e1 = 'L1 max'
            e3 = 'L3 max'
        elif switch == 'Min':
            new_L2.append(np.add(L2[i], e))
            new_L1.append(np.add(L1[i], e))
            new_L3.append(np.add(L3[i], e))
            e2 = f'{e}'
            e1 = f'{e}'
            e3 = f'{e}'

    L2 = np.array(new_L2)
    L1 = np.array(new_L1)
    L3 = np.array(new_L3)

# =======================================================
# score 계산

# score = np.add(L2, np.absolute(np.subtract(L1, L3)))
# score_func = '|L1 + L3| + L2'

# score = np.divide(L2, np.add(L1, L3))
# score_func = 'L2/(L1 + L3)'

# score = np.multiply(L2, np.add(1/L1, 1/L3))
# score_func = 'L2/(1/L1 + 1/L3)'

# score = np.add(np.divide(L2, np.add(L1, L3)), np.absolute(np.subtract(L1, L3)))
# score_func = 'L2/(L1 + L3 )+|L1 + L3|'

# score = np.multiply(L2, np.add(1/np.add(L1, epsilon), 1/np.add(L3, epsilon)))
# score_func = f'L2/(1/(L1 + e) + 1/(L3 + e)), e = {epsilon}'

# L1 = np.expand_dims(L1, axis=1)
# L3 = np.expand_dims(L3, axis=1)
# L13 = np.concatenate((L1, L3), axis=1)
# score = np.divide(L13, np.add(L2, epsilon))
# score_func = f'min(L1, L3)/(L2 + e), e = {epsilon}'

    if switch == 'Min':
        score = np.multiply(L2, np.add(1/L1, 1/L3))
        score_func = f'(L2 + e2)(1/(L1 + e1) + 1/(L3 + e3)), e1 = {e1}, e2 = {e2}, e3 = {e3}'
        # score 중 최소값 계산
        Min = []
        for i in range(num_all_data):
            Min.append(min(score[i]))
        Min = np.array(Min)

    elif switch == 'Max':
        L1 = np.expand_dims(L1, axis=1)
        L3 = np.expand_dims(L3, axis=1)
        L13 = np.concatenate((L1, L3), axis=1)
        temp_list = []
        for i in range(num_all_data):
            temp_list.append(np.min(L13[i], axis=0))
        L13 = np.array(temp_list)
        score = np.divide(L13, L2)
        score_func = f'(min(L1 + e1, L3 + e3))/(L2 + e2), e1 = {e1}, e2 = {e2}, e3 = {e3}'
        # score 중 최댓값 계산
        Max = []
        for i in range(num_all_data):
            Max.append(max(score[i]))
        Max = np.array(Max)
    elif switch == 'Multi':
        score = np.divide(np.multiply(np.add(L1, L3), 100), L2)
    else:
        print("Min 또는 Max 만 입력")
        exit()

    # =======================================================
    # score 최소값을 통해 entry point index 얻어내기
    entry = []
    for i in range(num_all_data):
        sur = data[i][:num_skin_points, :]
        if switch == 'Max':
            index = np.array(np.where(score[i] == Max[i]))
        elif switch == 'Min':
            index = np.array(np.where(score[i] == Min[i]))
        elif switch == 'Multi':
            index = np.array(np.where(score[i] == Max[i]))
        entry.append(sur[index[0][0]])

    entry = np.array(entry)


# ############################## recode time ###########################################################################
    print('\n')
    print('run time :', round(time.time() - start, 2), 'sec')


# ############################## make file #############################################################################
    np.save(f'{path}/score', score)
    np.save(f'{path}/score function type', switch)
    file = open(f'{path}/information.txt', 'a', encoding= 'utf8')
    file.write(f'\n\n\nscore function : {score_func}\n\n')
    print('\n\n추가 정보 기입')
    string = str(input())
    file.write(f'추가 정보: {string}')
    file.close()
    print('\n\n')