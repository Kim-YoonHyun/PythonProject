# _2
# word 설명에 맞추기, 반복 편의성 수정
# equalizing 이라는 단어를 전부 multi_targeting 으로 수정

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

    # type = sub, add
    multi_targeting_type_concat = 'concat'
    multi_targeting_type_sub = 'sub'
    multi_targeting_type_sub_concat = 'sub_concat'
    # multi_targeting_type_aver_sub = 'aver_sub'
    # multi_targeting_type_aver_concat = 'aver_concat'
    # multi_targeting_type_aver_sub_aver_concat = 'aver_sub_aver_concat'

    # ##############################
    data = np.load(f'{path}/data.npy')

    num_all_data = data.shape[0]
    num_all_points = data.shape[1]
    num_skin_points = np.load(f'{path}/num_skin_points.npy')
    num_bone1_points = np.load(f'{path}/num_bone1_points.npy')
    num_bone2_points = np.load(f'{path}/num_bone2_points.npy')

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
    np.save(f'{path}/MT_data_{multi_targeting_type_sub}', new_data_sub)

    # concat
    all_target_data_unlabel_tile = np.tile(all_target_data_unlabel, [1, num_all_points, 1])
    new_data_concat_unlabel = np.concatenate((all_data_unlabel, all_target_data_unlabel_tile), axis=2)
    new_data_concat = np.concatenate((new_data_concat_unlabel, all_label_data), axis=2)
    np.save(f'{path}/MT_data_{multi_targeting_type_concat}', new_data_concat)

    # sub, concat
    new_data_sub_concat_unlabel = np.concatenate((new_data_sub_unlabel, all_target_data_unlabel_tile), axis=2)
    new_data_sub_concat = np.concatenate((new_data_sub_concat_unlabel, all_label_data), axis=2)
    np.save(f'{path}/MT_data_{multi_targeting_type_sub_concat}', new_data_sub_concat)

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