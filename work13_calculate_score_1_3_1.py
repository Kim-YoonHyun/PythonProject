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

# 1_3_1
# 버전에 따른 추출 방식 적용 중


import numpy as np
import time
import os
import dill
import pandas as pd
# import functions_my_1_3 as fmy

def data_list_title(path):
    """
    class 객체를 불러와서 그 정보를 표시하는 함수
    :param path: './data'
    :return:
    """
    data_list = os.listdir(path)
    for idx, data in enumerate(data_list):
        array_size = np.load(f'{path}/{data}/input_data.npy').shape
        with open(f'{path}/{data}/data_information.pkl', 'rb') as file:
            _, bone1_data, _, _ = dill.load(file)

        print(f'{data}: {array_size}, [{bone1_data.rand_sam_status}, {bone1_data.trans_status}, {bone1_data.rot_status}]')


def euclidean_distance(start, end):
    """
    function of calculate the euclidean distance from start to end
    :param start: start vertex
    :param end: end vertex
    :return: euclidean distance
    """
    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
    return value


if __name__ == '__main__':
    input_data = np.load(f'data/data1/input_data.npy')
    data_df = pd.read_csv(f'data/data1/data_information.csv')
    print(data_df.columns)
    print(data_df['array_size'].value_counts())
    print(data_df['trans_status'].value_counts())
    print(data_df['rot_status'].value_counts())
    print(data_df['rand_sam_status'].value_counts())
    string = f'rand / 3, trans X 3, rot X 3, multi target X 5'
    exit()
    while True:
        # <title>------------------------------------------------------------
        # data list
        data_list_title(f'./data')
        exit()
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

        with open(f'{data_path}/data_information.pkl', 'rb') as file:
            target_data, bone1_data, bone2_data, skin_data = dill.load(file)

        # Bone1 및 Bone2 와 L1 간의 모든 거리 중 최소값(L2와 가장 가까운 Bone 좌표)구하기(L2, L3)
        L1 = euclidean_distance(start=skin_data.data_vertices_list, end=target_data.data_vertices_list)
        L2 = []
        L3 = []
        for idx, data in enumerate(input_data):
            print(f'{idx + 1}/{input_data.shape[0]}')
            L2.append(fmy.calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                                  np.array(bone1_data.data_vertices_list[idx]),
                                                                  np.array(skin_data.data_vertices_list[idx])))
            L3.append(fmy.calculate_point_to_line_length(np.array(target_data.data_vertices_list[idx]),
                                                                  np.array(bone2_data.data_vertices_list[idx]),
                                                                  np.array(skin_data.data_vertices_list[idx])))

        L1_e = np.add(L1, np.expand_dims(np.maximum.reduce(L1, axis=1), axis=1))
        L2_e = np.add(L2, np.expand_dims(np.maximum.reduce(L2, axis=1), axis=1))
        L3_e = np.add(L2, np.expand_dims(np.maximum.reduce(L3, axis=1), axis=1))

        if score_type == 'Max':
            score = np.divide(np.maximum(L2_e, L3_e), L1_e)
        else:
            score = np.divide(L1_e, np.maximum(L2_e, L3_e))


        print('run time :', round(time.time() - start, 2), 'sec')
        np.save(f'{data_path}/score', score)
        with open(f'{data_path}/score_type_{score_type}.txt', 'w') as file:
            file.write(score_type)
        # np.save(f'{data_path}/score function type={score_type}', score_type)
        print('\n')
    exit()

#
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