# _4
# 편의성 변경

import numpy as np
import os
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import Functions_mk2  # 오류 무시 가능


# #################################################### parameter #######################################################
test_percentage = 20

# #################################################### path setting ####################################################
model_list = os.listdir(f'./trained_model')
name_length_list = []
for i in range(len(model_list)):
    name_length_list.append(len(model_list[i]))
string_length = max(name_length_list)


while(True):
    # test 할 학습 모델 번호 입력
    Functions.model_list_title(file_path=f'./trained_model', row_num=3, string_length=string_length, blank_num=4)
    print(f'\nEnter the model number to calculate all information (1 ~ {len(model_list)})')
    model_num = int(input())

    # app 종료
    if model_num == 10000:
        break

    model_name = model_list[model_num - 1]
    print(model_name)
    train_path = f'./trained_model/{model_name}'
    test_path = f'./test_result/test_{model_name}'

    # path
    data_number = np.load(f'{train_path}/data_number.npy')
    data_path = f'../Spine/data/data{data_number}(Label)'

    # load array
    all_data = np.load(data_path + f'/data.npy')
    gt_data = np.load(data_path + f'/score.npy')
    num_skin_points = np.load(f'{data_path}/num_skin_points.npy')

    all_data_num = all_data.shape[0]
    all_points_num = all_data.shape[1]

    # test data
    test_data_num = round(all_data_num * (test_percentage / 100))
    test_data = all_data[all_data_num - test_data_num:, :, :]
    test_gt_data = gt_data[all_data_num - test_data_num:, :]

    # 계산용 기초 array 들
    i = 0
    all_D_list = np.array([[0], [0]])
    all_index_list = []
    all_loss_list = []
    all_test_output = np.zeros([1, num_skin_points])

    # =======================================================
    # D & loss 파일 내부의 모든 정보 불러오기
    while(True):
        try:
            index_k_loss = np.load(f'{test_path}/D & loss/{i + 1}_k_test_loss.npy')
            all_D = np.load(f'{test_path}/D & loss/{i + 1}_all_D.npy')
            test_output = np.expand_dims(np.load(f'{test_path}/D & loss/{i + 1}_test_output.npy'), 0)
        except FileNotFoundError:
            if i < test_data_num - 1:
                i += 1
                continue
            elif i > test_data_num - 1:
                break
        test_index = int(index_k_loss[0])
        k_value = int(index_k_loss[1]) - 1
        test_loss = index_k_loss[2]

        all_index_list.append(test_index)
        all_loss_list.append(test_loss)

        all_test_output = np.concatenate((all_test_output, test_output), axis=0)
        all_D_list = np.concatenate((all_D_list, all_D), axis= 1)
        i += 1

    all_test_output = np.delete(all_test_output, 1, axis=0)
    all_D_list = all_D_list[:, 1:]

    all_gt_score_min_vertex = []
    all_predicted_min_vertex = []
    all_top_k_aver_vertex = []
    # all_weighted_top_k_aver_vertex = []
    accuracy = []
    bar = []

    for i in range(all_test_output.shape[0]):
        skin = test_data[all_index_list[i]][:num_skin_points, 0:3]
        switch = np.load(f'{data_path}/score function type.npy')
        if switch == 'Min':
            optimal_score = min(test_gt_data[all_index_list[i]])
        elif switch == 'Max':
            optimal_score = max(test_gt_data[all_index_list[i]])

        optimal_index = np.array(np.where(test_gt_data[all_index_list[i]] == optimal_score))[0][0]

        # min vertex 계산
        gt_score_min_vertex = skin[optimal_index]
        all_gt_score_min_vertex.append(gt_score_min_vertex)

        # all_weighted_top_k_aver_vertex.append(Functions.Make_top_k_average_vertex(np.array([all_test_output[i]]), k_value, test_set, all_index_list[i], 0.1, 1))
        predicted_min_vertex = Functions_mk2.make_top_k_average_vertex_mk2(np.array(all_test_output[i]),
                                                                           test_data[:, :, 0:3][i], 1, 1, 1)
        top_k_aver_vertex = Functions_mk2.make_top_k_average_vertex_mk2(np.array(all_test_output[i]),
                                                                        test_data[:, :, 0:3][i], k_value, 1, 1)
        all_predicted_min_vertex.append(predicted_min_vertex)
        all_top_k_aver_vertex.append(top_k_aver_vertex)
        accuracy.append(round((1 - all_loss_list[i]) * 100, 2))
        bar.append(Functions.make_percentage_bar(round((1 - all_loss_list[i]) * 100, 1), 50))

    # loss 계산
    aver_loss = np.average(all_loss_list)
    max_loss = max(all_loss_list)
    min_loss = min(all_loss_list)

    # accuracy 계산
    aver_accuracy = round(np.average(np.array(accuracy)), 2)
    max_accuracy = max(accuracy)
    min_accuracy = min(accuracy)

    # all D 계산
    aver_pmv_D = np.average(all_D_list[0])
    aver_tkav_D = np.average(all_D_list[1])
    # aver_wtkav_D = np.average(all_D_list[2])
    max_pmv_D = max(all_D_list[0])
    max_tkav_D = max(all_D_list[1])
    # max_wtkav_D = max(all_D_list[2])
    min_pmv_D = min(all_D_list[0])
    min_tkav_D = min(all_D_list[1])
    # min_wtkav_D = min(all_D_list[2])

    # time 계산
    testing_time_array = np.load(f'{test_path}/testing_time_array.npy')
    aver_testing_time_array = round((np.average(testing_time_array)), 2)

    # 표준 편차 계산
    std = round(np.std(np.array(accuracy)), 2)

    # #################################################### 결과 저장 ###################################################
    # all test output 저장
    np.save(f'{test_path}/all_test_output', all_test_output)

    # max & min & average 값 통합저장.

    file = open(f'{test_path}/average D & loss.txt', 'w', encoding='utf8')
    line = f'average accuracy = {aver_accuracy} %\n' \
        f'average predicted min vertex D = {round(aver_pmv_D, 1)}\n' \
        f'average top {k_value} average vertex D = {round(aver_tkav_D, 1)}\n' \
        f'average testing time = {aver_testing_time_array} sec\n' \
        f'표준 편차 = {std}\n\n' \
        f'max accuracy = {max_accuracy} %, (test index: {1 + int(all_index_list[np.array(np.where(all_loss_list == max_loss))[0][0]])})\n' \
        f'min accuracy = {min_accuracy} %, (test index: {1 + int(all_index_list[np.array(np.where(all_loss_list == min_loss))[0][0]])})\n\n' \
        f'max pmv D = {round(max_pmv_D, 1)}, (test index: {1 + int(all_index_list[np.array(np.where(all_D_list[0] == max_pmv_D))[0][0]])})\n' \
        f'max tkav D = {round(max_tkav_D, 1)}, (test index: {1 + int(all_index_list[np.array(np.where(all_D_list[1] == max_tkav_D))[0][0]])})\n' \
        f'min pmv D = {round(min_pmv_D, 1)}, (test index: {1 + int(all_index_list[np.array(np.where(all_D_list[0] == min_pmv_D))[0][0]])})\n' \
        f'min tkav D = {round(min_tkav_D, 1)}, (test index: {1 + int(all_index_list[np.array(np.where(all_D_list[1] == min_tkav_D))[0][0]])})'
    file.write(line)
    file.close()

    file_test_info = open(f'{test_path}/Test information.txt', 'w', encoding='utf8')

    print(np.array(all_gt_score_min_vertex).shape)
    for i in range(all_test_output.shape[0]):
        line = f'test index : {all_index_list[i] + 1}\n' \
            f'predicted entry vertex :  [{round(all_predicted_min_vertex[i][0][0], 2)}, {round(all_predicted_min_vertex[i][0][1], 2)}, {round(all_predicted_min_vertex[i][0][2], 2)}]\n' \
            f'top {k_value} average vertex :  [{round(all_top_k_aver_vertex[i][0][0], 2)}, {round(all_top_k_aver_vertex[i][0][1], 2)}, {round(all_top_k_aver_vertex[i][0][2], 2)}]\n' \
            f'gt entry vertex:  [{round(all_gt_score_min_vertex[i][0], 2)}, {round(all_gt_score_min_vertex[i][1], 2)}, {round(all_gt_score_min_vertex[i][2], 2)}]\n' \
            f'pmv D : {round(all_D_list[0][i], 1)}\n' \
            f'tkav D : {round(all_D_list[1][i], 1)}\n' \
            f'testing time : {testing_time_array[i]} sec\n' \
            f'Test loss : {round(all_loss_list[i], 4)}\n' \
            f'{accuracy[i]} %  {bar[i]}\n\n'
        # f'weighted top {k_value} average vertex :  [{round(all_weighted_top_k_aver_vertex[i][0], 2)}, {round(all_weighted_top_k_aver_vertex[i][1], 2)}, {round(all_weighted_top_k_aver_vertex[i][2], 2)}]\n' \
        # f'wtkav D : {round(all_D_list[2][i], 1)}\n' \
        file_test_info.write(line)
    file_test_info.close()