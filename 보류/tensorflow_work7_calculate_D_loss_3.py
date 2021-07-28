# entry 를 test entry, train entry 로 구분
# 다른 test shape 를 받아들였을 떄의 dummy 데이터 읽기 추가.
# all_D_list 에서 3번쨰에 해당하는 weighted 값 제거

# _2_3
# 점점 느려지는 현상 수정, network 이름도 같이 저장

# _3
# word 설명에 맞추기
# skin point 및 score function type 는 데이터 폴더에서 직접 불러옴

import numpy as np
import os
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import Functions_mk2  # 오류 무시 가능


# ############################## range setting #########################################################################
max_model_num = len(os.walk('./test_result/').__next__()[1])


# ############################## Enter the parameter ###################################################################
while True:
    # test 할 학습 모델 번호 입력
    print(f'\nA. Enter the model number to calculate all information (1 ~ {max_model_num})')
    model_num = int(input())

    # app 종료
    if model_num == 10000:
        break

    # 제대로 불러졌는지 확인용
    model_name = os.listdir(f"./training_model")[model_num-1]
    print(model_name)

    # path 설정
    train_path = f'./training_model/{model_name}'
    test_path = f'./test_result/test_{model_name}'
    data_number = np.load(f'{train_path}/data_number.npy')
    data_path = f'./data/data{data_number}(Label)'

    num_skin_points = np.load(f'{data_path}/num_skin_points.npy')
    test_GT_data = np.load(f'{train_path}/test_GT_data.npy')
    test_data = np.load(f'{train_path}/test_data.npy')
    num_test_data = test_data.shape[0]

    # 계산용 기초 array 들
    i = 0
    all_D_list = np.array([[0], [0]])
    all_index_list = []
    all_loss_list = []
    all_test_output = np.array([[]])

    # =======================================================
    # D & loss 파일 내부의 모든 정보 불러 오기
    while True:
        try:
            index_k_loss = np.load(f'{test_path}/D & loss/{i}_k_test_loss.npy')
            all_D = np.load(f'{test_path}/D & loss/{i}_all_D.npy')
            test_output = np.load(f'{test_path}/D & loss/{i}_test_output.npy')
        except FileNotFoundError:
            if i < num_test_data:
                i += 1
                continue
            elif i == num_test_data:
                break

        test_index = int(index_k_loss[0])
        k_value = int(index_k_loss[1]) - 1
        test_loss = index_k_loss[2]

        all_index_list.append(test_index)
        all_loss_list.append(test_loss)

        if i == 0:
            all_test_output = np.concatenate((all_test_output, test_output), axis=1)
        else:
            all_test_output = np.concatenate((all_test_output, test_output), axis=0)
        all_D_list = np.concatenate((all_D_list, all_D), axis= 1)
        i += 1

    all_D_list = all_D_list[:, 1:]

    all_gt_score_min_vertex = []
    all_predicted_min_vertex = []
    all_top_k_aver_vertex = []
    # all_weighted_top_k_aver_vertex = []
    accuracy = []
    bar = []

    for i in range(all_test_output.shape[0]):
        skin = test_data[all_index_list[i]][:num_skin_points, 0:3]
        test_GT_data_skin = test_GT_data[all_index_list[i]][0:num_skin_points]
        switch = np.load(f'{data_path}/score function type.npy')
        if switch == 'Min':
            optimal_score = min(test_GT_data_skin)
        elif switch == 'Max':
            optimal_score = max(test_GT_data_skin)

        optimal_index = np.array(np.where(test_GT_data_skin == optimal_score))[0][0]

        # min vertex 계산
        gt_score_min_vertex = skin[optimal_index]
        all_gt_score_min_vertex.append(gt_score_min_vertex)

        if switch == 'Min':
            predicted_min_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(np.array(all_test_output[i]),
                                                                           test_data[:, :, 0:3][i], 1, 1, 1)
            top_k_aver_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(np.array(all_test_output[i]),
                                                                        test_data[:, :, 0:3][i], k_value, 1, 1)
        elif switch == 'Max':
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
    max_pmv_D = max(all_D_list[0])
    max_tkav_D = max(all_D_list[1])
    min_pmv_D = min(all_D_list[0])
    min_tkav_D = min(all_D_list[1])

    # time 계산
    testing_times = np.load(f'{test_path}/testing_times.npy')
    aver_testing_times = round((np.average(testing_times)), 2)

    # 표준 편차 계산
    std_acc = round(np.std(np.array(accuracy)), 2)
    std_D1 = round(np.std(np.array(all_predicted_min_vertex)), 2)
    std_D2 = round(np.std(np.array(all_top_k_aver_vertex)), 2)


    ##################################################### 결과 저장 ########################################################
    # all test output 저장
    np.save(f'{test_path}/all_test_output', all_test_output)

    # max & min & average 값 통합저장.

    file = open(f'{test_path}/average D & loss.txt', 'w', encoding='utf8')
    line = f'average accuracy = {aver_accuracy} %\n' \
        f'average predicted min vertex D = {round(aver_pmv_D, 1)}\n' \
        f'average top {k_value} average vertex D = {round(aver_tkav_D, 1)}\n' \
        f'average testing time = {aver_testing_times} sec\n' \
        f'정확도 표준 편차 = {std_acc}\n' \
        f'D1 표준 편차 = {std_D1}\n' \
        f'D2 표준 편차 = {std_D2}\n\n' \
        f'max accuracy = {max_accuracy} %, (test index: {int(all_index_list[np.array(np.where(all_loss_list == max_loss))[0][0]])})\n' \
        f'min accuracy = {min_accuracy} %, (test index: {int(all_index_list[np.array(np.where(all_loss_list == min_loss))[0][0]])})\n\n' \
        f'max pmv D = {round(max_pmv_D, 1)}, (test index: {int(all_index_list[np.array(np.where(all_D_list[0] == max_pmv_D))[0][0]])})\n' \
        f'max tkav D = {round(max_tkav_D, 1)}, (test index: {int(all_index_list[np.array(np.where(all_D_list[1] == max_tkav_D))[0][0]])})\n' \
        f'min pmv D = {round(min_pmv_D, 1)}, (test index: {int(all_index_list[np.array(np.where(all_D_list[0] == min_pmv_D))[0][0]])})\n' \
        f'min tkav D = {round(min_tkav_D, 1)}, (test index: {int(all_index_list[np.array(np.where(all_D_list[1] == min_tkav_D))[0][0]])})' \


    file.write(line)
    file.close()

    # 각 test 별 정보 저장
    file_test_info = open(f'{test_path}/Test information.txt', 'w', encoding='utf8')
    for i in range(all_test_output.shape[0]):
        line = f'test index : {all_index_list[i]}\n' \
            f'predicted entry vertex :  [{round(all_predicted_min_vertex[i][0][0], 2)}, {round(all_predicted_min_vertex[i][0][1], 2)}, {round(all_predicted_min_vertex[i][0][2], 2)}]\n' \
            f'top {k_value} average vertex :  [{round(all_top_k_aver_vertex[i][0][0], 2)}, {round(all_top_k_aver_vertex[i][0][1], 2)}, {round(all_top_k_aver_vertex[i][0][2], 2)}]\n' \
            f'gt entry vertex:  [{round(all_gt_score_min_vertex[i][0], 2)}, {round(all_gt_score_min_vertex[i][1], 2)}, {round(all_gt_score_min_vertex[i][2], 2)}]\n' \
            f'pmv D : {round(all_D_list[0][i], 1)}\n' \
            f'tkav D : {round(all_D_list[1][i], 1)}\n' \
            f'testing time : {testing_times[i]} sec\n' \
            f'Test loss : {round(all_loss_list[i], 4)}\n' \
            f'{accuracy[i]} %  {bar[i]}\n\n'
        file_test_info.write(line)
    file_test_info.close()