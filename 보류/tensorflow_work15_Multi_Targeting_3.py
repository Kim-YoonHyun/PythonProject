# _2 >>> 20210618 지금까지 모든 결과를 추출한 코드
# word 설명에 맞추기, 반복 편의성 수정
# equalizing 이라는 단어를 전부 multi_targeting 으로 수정

# _3
# 데이터를 선택할때 폴더 최대갯수가 아닌 폴더 이름을 통해 범위를 설정
# data 는 기본적으로 labeled 된 데이터만 불러오도록 변경

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
    print('data number for make Multi Target')
    print(folder_number_list)
    
    # 입력 번호가 list에 없을 경우 오류 처리
    try:
        index = folder_number_list.index(int(input('Enter the data number : ')))
        data_number = folder_number_list[index]
    except ValueError:
        print('리스트에 없는 번호입니다.')
        continue

    # data path 설정
    data_path = './data/data' + str(data_number)

    # 입력 번호 데이터가 이미 MT 계산이 완료된 경우 처리
    if os.path.isfile(data_path + '/MT.npy'):
        print('\n\n\n이미 MT 계산이 완료된 데이터입니다.')
        continue

# ############################## path 설정 #############################################################################
    # type = sub, add
    multi_targeting_type_concat = 'concat'
    multi_targeting_type_sub = 'sub'
    multi_targeting_type_sub_concat = 'sub_concat'
    # multi_targeting_type_aver_sub = 'aver_sub'
    # multi_targeting_type_aver_concat = 'aver_concat'
    # multi_targeting_type_aver_sub_aver_concat = 'aver_sub_aver_concat'

    data = np.load(data_path + '/data_labeled.npy')
    num_all_data = data.shape[0]
    num_all_points = data.shape[1]
    num_skin_points = np.load(f'{data_path}/num_skin_points.npy')
    num_bone1_points = np.load(f'{data_path}/num_bone1_points.npy')
    num_bone2_points = np.load(f'{data_path}/num_bone2_points.npy')

    all_data_unlabel = data[:, :, 0:3]

    all_skin_data_unlabel = all_data_unlabel[:, 0:num_skin_points, :]
    print(f'all skin   : {all_skin_data_unlabel.shape}')
    all_bone1_data_unlabel = all_data_unlabel[:, num_skin_points:num_skin_points + num_bone1_points, :]
    print(f'all bone1  : {all_bone1_data_unlabel.shape}')
    all_bone2_data_unlabel = all_data_unlabel[:, num_skin_points + num_bone1_points:num_skin_points + num_bone1_points + num_bone2_points, :]
    print(f'all bone2  : {all_bone2_data_unlabel.shape}')
    all_bone_data_unlabel = all_data_unlabel[:, num_skin_points:num_skin_points + num_bone1_points + num_bone2_points, 0:3]
    print(f'all bone   : {all_bone_data_unlabel.shape}')
    all_target_data_unlabel = all_data_unlabel[:, num_all_points - 1:, :]
    print(f'all target : {all_target_data_unlabel.shape}\n')

    all_label_data = data[:, :, 3:]
    print(f'all label : {all_label_data.shape}\n')
    all_data_except_target_unlabel = all_data_unlabel[:, 0:num_all_points - 1, :]
    print(f'all data (no target) : {all_data_except_target_unlabel.shape}')

    # subtract
    # all_target_data_tile = np.tile(all_target_data, [1, num_all_points - 1, 1])
    new_data_sub_except_target_unlabel = np.subtract(all_data_except_target_unlabel, all_target_data_unlabel)
    new_data_sub_unlabel = np.concatenate((new_data_sub_except_target_unlabel, all_target_data_unlabel), axis=1)
    new_data_sub = np.concatenate((new_data_sub_unlabel, all_label_data), axis=2)
    np.save(f'{data_path}/MT_data_{multi_targeting_type_sub}', new_data_sub)

    # concat
    all_target_data_unlabel_tile = np.tile(all_target_data_unlabel, [1, num_all_points, 1])
    new_data_concat_unlabel = np.concatenate((all_data_unlabel, all_target_data_unlabel_tile), axis=2)
    new_data_concat = np.concatenate((new_data_concat_unlabel, all_label_data), axis=2)
    np.save(f'{data_path}/MT_data_{multi_targeting_type_concat}', new_data_concat)

    # sub, concat
    new_data_sub_concat_unlabel = np.concatenate((new_data_sub_unlabel, all_target_data_unlabel_tile), axis=2)
    new_data_sub_concat = np.concatenate((new_data_sub_concat_unlabel, all_label_data), axis=2)
    np.save(f'{data_path}/MT_data_{multi_targeting_type_sub_concat}', new_data_sub_concat)

    # MT 됬는지 확인용
    MT = np.array(['MT'])
    np.save(data_path + '/MT', MT)

    print('\n\n\n\n')
    # # aver sub
    # all_target_data_unlabel_aver = np.expand_dims(np.average(all_target_data_unlabel, axis=2), axis=1)
    # new_data_aver_sub_except_target_unlabel = np.subtract(all_data_except_target_unlabel, all_target_data_unlabel_aver)
    # new_data_aver_sub_unlabel = np.concatenate((new_data_aver_sub_except_target_unlabel, all_target_data_unlabel), axis=1)
    # new_data_aver_sub = np.concatenate((new_data_aver_sub_unlabel, all_label_data), axis=2)
    # np.save(f'{data_path}/MT_data_{multi_targeting_type_aver_sub}', new_data_aver_sub)
    #
    # # aver concat
    # all_target_data_unlabel_aver_tile = np.tile(all_target_data_unlabel_aver, [1, num_all_points, 1])
    # new_data_aver_concat_unlabel = np.concatenate((all_data_unlabel, all_target_data_unlabel_aver_tile), axis=2)
    # new_data_aver_concat = np.concatenate((new_data_aver_concat_unlabel, all_label_data), axis=2)
    # np.save(f'{data_path}/MT_data_{multi_targeting_type_aver_concat}', new_data_aver_concat)
    #
    # # aver sub aver concat
    # new_data_aver_sub_aver_concat_unlabel = np.concatenate((new_data_aver_sub_unlabel, all_target_data_unlabel_aver_tile), axis=2)
    # new_data_aver_sub_aver_concat = np.concatenate((new_data_aver_sub_aver_concat_unlabel, all_label_data), axis=2)
    # np.save(f'{data_path}/MT_data_{multi_targeting_type_aver_sub_aver_concat}', new_data_aver_sub_aver_concat)

    # bone sub