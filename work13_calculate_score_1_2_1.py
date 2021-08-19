# _2
# 복잡한 계산 간결화

# _3
# score 최적점을 0 에서 1 로 변경
# _1
# epsilon 을 L1, L2, L3 각각의 최댓값으로 설정함

# _4
# word 설명과 맞추기
# _4_1
# 실행전후 파라미터 변경, 편의성 수정
# Functions import 방식 변경

# _5
# 코드 renewal
# pandas, class 등 추가 사용
# data_list_title 함수 사용

import numpy as np
import time
import os
import dill
import pandas as pd
import functions_my

if __name__ == '__main__':
    while True:
        # <title>------------------------------------------------------------
        # data list
        functions_my.data_list_title(f'./data')
        data_num = int(input('\n 계산할 데이터 번호 입력 > '))
        if data_num == 10000:
            break

        switch = int(input('\nscore type 선택. 1: Max, 2: Min > '))
        if switch == 1:
            score_type = 'Max'
        else:
            score_type = 'Min'

        # <score calculate>---------------------------------------------------------------------
        print(f'\nscore calculate...')
        start = time.time()
        data_path = f'./data/data{data_num}'
        input_data = np.load(f'{data_path}/input_data.npy')
        if data_num < 6:
            with open(f'{data_path}/input_data.pkl', 'rb') as file:
                target_data, bone1_data, bone2_data, skin_data = dill.load(file)
        else:
            with open(f'{data_path}/data_information.pkl', 'rb') as file:
                target_data, bone1_data, bone2_data, skin_data = dill.load(file)

        print(input_data.shape)

        L1 = functions_my.euclidean_distance(start=skin_data.data_vertices_list, end=target_data.data_vertices_list)

        # Bone1 및 Bone2 와 L2 간의 모든 거리 중 최소값(L2와 가장 가까운 Bone 좌표)구하기(L1, L3)
        L2 = []
        L3 = []
        for idx, data in enumerate(input_data):
            print(f'{idx + 1}/{input_data.shape[0]}')
            L2.append(functions_my.calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                                  np.array(bone1_data.data_vertices_list[idx]),
                                                                  np.array(skin_data.data_vertices_list[idx])))
            L3.append(functions_my.calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                                  np.array(bone2_data.data_vertices_list[idx]),
                                                                  np.array(skin_data.data_vertices_list[idx])))

        L1_e = np.add(L1, np.expand_dims(np.maximum.reduce(L1, axis=1), axis=1))
        L2_e = np.add(L2, np.expand_dims(np.maximum.reduce(L2, axis=1), axis=1))
        L3_e = np.add(L2, np.expand_dims(np.maximum.reduce(L3, axis=1), axis=1))

        if score_type == 'Max':
            score = np.divide(np.maximum(L2_e, L3_e), L1_e)
        else:
            score = np.divide(L1_e, np.maximum(L2_e, L3_e))

        print('\n')
        print('run time :', round(time.time() - start, 2), 'sec')
        np.save(f'{data_path}/score', score)
        with open(f'{data_path}/score_type_{score_type}.txt', 'w') as file:
            file.write(score_type)
        # np.save(f'{data_path}/score function type={score_type}', score_type)
    exit()

# # ############################## parameter ##############################################################
# while True:
# #     max_data_num = len(os.walk('./data/').__next__()[1])
# #     print(f'A. enter the data number for calculate score 1 ~ {max_data_num}')
# #     data_num = int(input())
# #     print(f'B. enter the score function type ( Max or Min )')
# #     switch = str(input())
# #     if switch == 'Min':
# #         print(f'B. enter the epsilon value')
# #         e = int(input())
# #         print(e)
# #
# #     start = time.time()
# #     path = f'./data/data{data_num}'
# # # #################################################### calculate #######################################################
# #     # data 불러오기
# #     data = np.load(f'{path}/data.npy')
# #     num_skin_points = np.load(f'{path}/num_skin_points.npy')
# #     num_B1_points = np.load(f'{path}/num_bone1_points.npy')
# #     num_B2_points = np.load(f'{path}/num_bone2_points.npy')
# #     num_tar_points = 1
# #
# #     print('\nnumber of data:', data.shape[0])
# #     print('number of Skin points:', num_skin_points)
# #     print('number of Bone1 points:', num_B1_points)
# #     print('number of Bone2 points:', num_B2_points)
# #
# #     num_all_data = data.shape[0]
# #
# #     # =======================================================
# #     # surface 와 target 간의 모든 거리 구하기(L2)
# #     skin = data[:num_all_data, 0:num_skin_points, :]
# #     Bone1 = data[:num_all_data, num_skin_points:num_skin_points+num_B1_points, :]
# #     Bone2 = data[:num_all_data, num_skin_points+num_B1_points:num_skin_points+num_B1_points+num_B2_points, :]
# #     tar = data[:num_all_data, num_skin_points+num_B1_points+num_B2_points:num_skin_points+num_B1_points+num_B2_points+num_tar_points, :]
# #     L2 = Functions.euclidean_distance(start=skin, end=tar)
# #
# #     # =======================================================
# #     # Bone1 및 Bone2 와 L2 간의 모든 거리 중 최소값(L2와 가장 가까운 Bone 좌표)구하기(L1, L3)
# #     L1 = []
# #     L3 = []
# #     for idx in range(num_all_data):
# #         print(f'{idx + 1}/{num_all_data}')
# #         L1.append(functions.calculate_point_to_line_length(tar[idx], Bone1[idx], skin[idx]))
# #         L3.append(Functions.calculate_point_to_line_length(tar[idx], Bone2[idx], skin[idx]))
# #     L1 = np.array(L1)
# #     L3 = np.array(L3)
#
# #     new_L2 = []
# #     new_L1 = []
# #     new_L3 = []
# #
# #     for i in range(input_data.shape[0]):
# #         if score_type == 'Max':
# #             new_L2.append(np.add(L2[i], max(L2[i])))
# #             new_L1.append(np.add(L1[i], max(L1[i])))
# #             new_L3.append(np.add(L3[i], max(L3[i])))
# #             e2 = 'L2 max'
# #             e1 = 'L1 max'
# #             e3 = 'L3 max'
# #         elif score_type == 'Min':
# #             new_L2.append(np.add(L2[i], e))
# #             new_L1.append(np.add(L1[i], e))
# #             new_L3.append(np.add(L3[i], e))
# #             e2 = f'{e}'
# #             e1 = f'{e}'
# #             e3 = f'{e}'
# #
# #     L2 = np.array(new_L2)
# #     L1 = np.array(new_L1)
# #     L3 = np.array(new_L3)
# #
# # # =======================================================
# # # score 계산
# #
# # # score = np.add(L2, np.absolute(np.subtract(L1, L3)))
# # # score_func = '|L1 + L3| + L2'
# #
# # # score = np.divide(L2, np.add(L1, L3))
# # # score_func = 'L2/(L1 + L3)'
# #
# # # score = np.multiply(L2, np.add(1/L1, 1/L3))
# # # score_func = 'L2/(1/L1 + 1/L3)'
# #
# # # score = np.add(np.divide(L2, np.add(L1, L3)), np.absolute(np.subtract(L1, L3)))
# # # score_func = 'L2/(L1 + L3 )+|L1 + L3|'
# #
# # # score = np.multiply(L2, np.add(1/np.add(L1, epsilon), 1/np.add(L3, epsilon)))
# # # score_func = f'L2/(1/(L1 + e) + 1/(L3 + e)), e = {epsilon}'
# #
# # # L1 = np.expand_dims(L1, axis=1)
# # # L3 = np.expand_dims(L3, axis=1)
# # # L13 = np.concatenate((L1, L3), axis=1)
# # # score = np.divide(L13, np.add(L2, epsilon))
# # # score_func = f'min(L1, L3)/(L2 + e), e = {epsilon}'
# #     if switch == 'Min':
# #         score = np.multiply(L2, np.add(1/L1, 1/L3))
# #         score_func = f'(L2 + e2)(1/(L1 + e1) + 1/(L3 + e3)), e1 = {e1}, e2 = {e2}, e3 = {e3}'
# #         # score 중 최소값 계산
# #         Min = []
# #         for i in range(num_all_data):
# #             Min.append(min(score[i]))
# #         Min = np.array(Min)
# #
# #     elif switch == 'Max':
# #         L1 = np.expand_dims(L1, axis=1)
# #         L3 = np.expand_dims(L3, axis=1)
# #         L13 = np.concatenate((L1, L3), axis=1)
# #         temp_list = []
# #         for i in range(num_all_data):
# #             temp_list.append(np.min(L13[i], axis=0))
# #         L13 = np.array(temp_list)
# #         score = np.divide(L13, L2)
# #         score_func = f'(min(L1 + e1, L3 + e3))/(L2 + e2), e1 = {e1}, e2 = {e2}, e3 = {e3}'
# #         # score 중 최댓값 계산
# #         Max = []
# #         for i in range(num_all_data):
# #             Max.append(max(score[i]))
# #         Max = np.array(Max)
# #     elif switch == 'Multi':
# #         score = np.divide(np.multiply(np.add(L1, L3), 100), L2)
# #     else:
# #         print("Min 또는 Max 만 입력")
# #         exit()
#
#     # # =======================================================
#     # # score 최소값을 통해 entry point index 얻어내기
#     # entry = []
#     # for i in range(num_all_data):
#     #     sur = data[i][:num_skin_points, :]
#     #     if switch == 'Max':
#     #         index = np.array(np.where(score[i] == Max[i]))
#     #     elif switch == 'Min':
#     #         index = np.array(np.where(score[i] == Min[i]))
#     #     elif switch == 'Multi':
#     #         index = np.array(np.where(score[i] == Max[i]))
#     #     entry.append(sur[index[0][0]])
#     #
#     # entry = np.array(entry)
#
#
# # ############################## recode time ###########################################################################
#     print('\n')
#     print('run time :', round(time.time() - start, 2), 'sec')
#
#
# # ############################## make file #############################################################################
#     np.save(f'{path}/score', score)
#     np.save(f'{path}/score function type', switch)
#     file = open(f'{path}/information.txt', 'a', encoding= 'utf8')
#     file.write(f'\n\n\nscore function : {score_func}\n\n')
#     print('\n\n추가 정보 기입')
#     string = str(input())
#     file.write(f'추가 정보: {string}')
#     file.close()
#     print('\n\n')