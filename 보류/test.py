# _2
# validation 제대로 정의
# _2_1
# 변수 재설정, shuffle data 사용 추가

# _3
# dummy 사용 여부 추가
# _3_1
# 모델 저장시 더 많은 정보 저장

# _4
# dummy 관련 삭제
# _4_1
# scheduler 를 통해서 learning rate 점차적으로 감소

# _5
# 저장되는 내용 및 이름 변경
# _5_1
# Macro_7 과 연동
# _5_2
# 저장되는 파일 추가
# _5_2_1
# 저장되는 명칭 변경
# _5_3
# Multi Targeting 추가
# _5_4
# edge feature 관련 추가
# _6
# 코드 전반적인 정리
# edgeConv 사용여부 결정
# data path 를 tensorflow 프로젝트 폴더에서 직접 불러오는 방식으로 변경
# _7
# training 과정에서 dummy 제거


import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import os
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import Functions_mk2  # 오류 무시 가능
import matplotlib.pyplot as plt
# import EdgeConv.Net_2_seq_3_ch_128_Ker_1 as network
import DGCNN_Network.Net_2_Seq_1_Ch_32_Ker_3 as network

# import pyautogui as m

torch.manual_seed(289343610626200)

# GPU 세팅
use_cuda = torch.cuda.is_available()
print('GPU 사용 가능 여부:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
# print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
# print('graphic name:', torch.cuda.get_device_name())
cuda = torch.device('cuda')
print(cuda)
# print('cuda seed: ', torch.cuda.initial_seed())


# ######################################## parameter ###################################################################
# data number
data_number = 2

# 비율 (백분율 기준)
test_percentage = 20
val_percentage = 20

# Multi_Targeting 여부
Multi_Targeting = 'sub'

# epochs
epochs = 4

# validation epoch
val_epoch = 2

# learning rate
learning_rate = 0.1
gamma = 0.84

# EdgeConv parameter
edge = 'off'

# code testing 여부
code_testing = 'on'

# Macro 사용 여부
macro = 'oan'

# ######################################## data setting ################################################################
# path
data_path = f'./data/data{data_number}(Label)'

# load data
if Multi_Targeting == 'sub':
    all_data = np.load(data_path + f'/MT_data_sub.npy')
elif Multi_Targeting == 'concat':
    all_data = np.load(data_path + f'/MT_data_concat.npy')
elif Multi_Targeting == 'sub_concat':
    all_data = np.load(data_path + f'/MT_data_sub_concat.npy')
else:
    all_data = np.load(data_path + f'/data.npy')

gt_data = np.load(data_path + f'/score_add_dummy.npy')
num_skin_points = np.load(data_path + f'/num_skin_points.npy')

# variables
all_data_num = all_data.shape[0]
all_points_num = all_data.shape[1]
test_data_num = round(all_data_num * (test_percentage/100))
val_data_num = round((all_data_num - test_data_num) * (val_percentage/100))
train_data_num = all_data_num - test_data_num - val_data_num

# train/val set separate
train_data = all_data[:train_data_num, :, :]
val_data = all_data[train_data_num:train_data_num + val_data_num]

# gt train/val set
train_gt_data = gt_data[:train_data_num]
val_gt_data = gt_data[train_data_num:train_data_num + val_data_num]

# print
print('\n')
print('size of all data       : ', all_data.shape)
print('size of train data     : ', train_data.shape)
print('size of validation data: ', val_data.shape)
print('\n')
print('size of gt data (score)     : ', gt_data.shape)
print('size of train gt data       : ', train_gt_data.shape)
print('size of validation gt data  : ', val_gt_data.shape)


# ############################## data 전처리 ###########################################################################
# normalize
norm_train_data = Functions_mk2.normalize_data(train_data, num_skin_points, all_points_num)
norm_train_gt_data = Functions_mk2.normalize_data(train_gt_data, num_skin_points, all_points_num)
norm_val_data = Functions_mk2.normalize_data(val_data, num_skin_points, all_points_num)
norm_val_gt_data = Functions_mk2.normalize_data(val_gt_data, num_skin_points, all_points_num)

# numpy 를 torch 로 변환
norm_train_data = torch.from_numpy(norm_train_data).float()
norm_train_gt_data = torch.from_numpy(norm_train_gt_data).float()
norm_val_data = torch.from_numpy(norm_val_data).float()
norm_val_gt_data = torch.from_numpy(norm_val_gt_data).float()

# GPU 설정
if use_cuda:
    norm_train_data = norm_train_data.to(cuda)
    norm_train_gt_data = norm_train_gt_data.to(cuda)
    norm_val_data = norm_val_data.to(cuda)
    norm_val_gt_data = norm_val_gt_data.to(cuda)


# ############################## training ##############################################################################
# network
net = network.Net()
if use_cuda:
    net = net.to(cuda)
print(net)

# loss function
criterion = nn.MSELoss(reduction='mean')

# optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=val_epoch, gamma=gamma)

# train & val loss list
train_aver_loss_list = []
train_aver_accuracy_list = []
val_aver_loss_list = []
val_aver_accuracy_list = []

# time stamp
start = time.time()

for epoch in range(epochs):

    scheduler.step()

    train_loss_list = []
    val_loss_list = []

    epoch = epoch + 1
    train_loss = 0.0

    for idx, input_data in enumerate(norm_train_data):
        if code_testing == 'on':
            if idx == 2:
                break
        if edge != 'on':
            input_data = torch.transpose(input_data, 0, 1)
            input_data = torch.unsqueeze(input_data, 0)
            input_data = torch.unsqueeze(input_data, 2)


        # grad init
        optimizer.zero_grad()

        # forward propagation
        model_output = net(input_data)

        model_output = model_output[0:num_skin_points]
        norm_train_gt_data = norm_train_gt_data[:, 0:num_skin_points]
        # calculate loss
        loss = criterion(model_output, norm_train_gt_data[idx])

        train_loss = loss.item()**0.5
        print(train_loss)
        exit()
        # train_loss = loss.item()
        train_loss_list.append(train_loss)

        # back propagation
        loss.backward()

        # weight update
        optimizer.step()

        del loss
        del model_output

    train_loss_array = np.array(train_loss_list)

    # average loss
    train_aver_loss = np.average(train_loss_array)
    train_aver_loss_list.append(train_aver_loss)

    # average accuracy
    train_aver_accuracy = (1 - train_aver_loss)*100
    train_aver_accuracy_list.append(train_aver_accuracy)

    # print
    bar = Functions.make_percentage_bar(train_aver_accuracy, 50)
    print(f'%3d/{epochs}, {scheduler.optimizer.state_dict()["param_groups"][0]["lr"]}, {bar} train aver loss: %.5f -> aver accuracy: %.2f' %(epoch, train_aver_loss, train_aver_accuracy) + " %")

    # validation
    if epoch % val_epoch == 0:
        val_loss = 0
        for jdx, val_input in enumerate(norm_val_data):
            if code_testing == 'on':
                if jdx == 2:
                    break

            if edge != 'on':
                val_input = torch.transpose(val_input, 0, 1)
                val_input = torch.unsqueeze(val_input, 0)
                val_input = torch.unsqueeze(val_input, 2)

            val_output = net(val_input)
            val_loss = criterion(val_output, norm_val_gt_data[jdx]).item()**0.5
            val_loss_list.append(val_loss)

        val_loss_array = np.array(val_loss_list)

        val_aver_loss = np.average(val_loss_array)
        val_aver_loss_list.append(val_aver_loss)

        val_aver_accuracy = (1 - val_aver_loss)*100
        val_aver_accuracy_list.append(val_aver_accuracy)

        bar = Functions.make_percentage_bar(val_aver_accuracy, 50)
        print(f'\n %3d/{epochs}, {scheduler.optimizer.state_dict()["param_groups"][0]["lr"]}, {bar} val aver loss: %.5f -> %.2f' % (epoch, val_aver_loss, val_aver_accuracy) + "%\n")

    # epoch graph
    if epoch % val_epoch == 0:
        plt.plot(epoch, train_aver_loss, 'b.')
        plt.plot(epoch, val_aver_loss, 'r.')
    else:
        plt.plot(epoch, train_aver_loss, 'b.')
    plt.pause(0.1)

plt.plot(epochs, train_aver_loss, label='train loss')
plt.plot(epochs, val_aver_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.title('epochs graph')

training_time, hour, minute, sec = Functions.record_time(start)
print(f'\nTraining time : {training_time} = {hour}시간 {minute}분 {sec}초\n\n')

train_aver_loss_array = np.array(train_aver_loss_list)
train_aver_accuracy_array = np.array(train_aver_accuracy_list)
v_aver_loss_array = np.array(val_aver_loss_list)
v_aver_accuracy_array = np.array(val_aver_accuracy_list)


# ####################################### save ########################################################################
# model name
network_name_list = os.path.basename(os.path.abspath(network.__file__)).split('.')[0].split('_')

# network_name_len = len(network_name_list)
network_name_len = 2

line = ''
for i in range(network_name_len):
    line += network_name_list[i]
    if i != (network_name_len - 1):
        line += '_'

model_name = line
# model_name = f'data_{data_number}_epochs_{epochs}'

# model number
max_model_num = len(os.walk('./trained_model/').__next__()[1]) + 1

# data type
data_type = '(L)'

# path setting
if max_model_num < 10:
    save_path = f'./trained_model/0{max_model_num}_model_{model_name}_epochs_{epochs}_data_{data_number}_loss_{criterion}'
else:
    save_path = f'./trained_model/{max_model_num}_model_{model_name}_epochs_{epochs}_data_{data_number}_loss_{criterion}'

# 폴더 생성
os.makedirs(f'{save_path}')

# save epoch graph
fig = plt.gcf()  # 변경한 곳
fig.savefig(f'{save_path}/epoch gragh.png')
plt.close(fig=fig)

# save trained model
torch.save(net, f'{save_path}/trained_model.pth')
torch.save(net.state_dict(), f'{save_path}/model_state_dict.pth')
torch.save({
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict()}, f'{save_path}/all.tar')

# save data information
# np.save(f'{save_path}/num_skin_points', num_skin_points)
np.save(f'{save_path}/data_number', data_number)

# save training information
file = open(f'{save_path}/training information.txt', 'w', encoding='utf8')
line = f'< {model_name} >\n\n' \
    f'CNN sequence : {network_name_list[3]}\n' \
    f'CNN channels : {network_name_list[5]}\n' \
    f'Kernel size : [{network_name_list[7]}, {network_name_list[7]}]\n\n' \
    f'Loss type : {criterion}\n' \
    f'Data number : {data_number}\n' \
    f'Data : {data_type}\n\n' \
    f'Epochs : {epochs}\n' \
    f'Validation epoch : {val_epoch}\n\n' \
    f'Start learning rate : {learning_rate}\n' \
    f'Gamma(step = val) : {gamma}\n' \
    f'Final learning rate : {scheduler.optimizer.state_dict()["param_groups"][0]["lr"]}\n\n' \
    f'Train data num : {train_data_num} ({100 - test_percentage}% / {100 - val_percentage}%)\n' \
    f'Validation num : {val_data_num} ({100 - test_percentage}% / {val_percentage}%)\n\n' \
    f'Training time : {training_time} = {hour}시간 {minute}분 {sec}초\n\n\n' \
    f'< Accuracy per epoch >\n'
file.write(line)

for i in range(epochs):
    file.write(f'train aver loss after epoch {i + 1}: %.5f -> %.2f' % (train_aver_loss_array[i],
                                                                       train_aver_accuracy_array[i]) + " %\n")
    if (i+1) % val_epoch == 0:
        j = ((i + 1) // val_epoch) - 1
        file.write(f'\n validation aver loss after epoch {i + 1}: %.5f -> %.2f' %(v_aver_loss_array[j],
                                                                                  v_aver_accuracy_array[j]) + " %\n\n")
file.close()

# test path setting
if max_model_num < 10:
    test_path = f'./test_result/test_0{max_model_num}_model_{model_name}_epochs_{epochs}_data_{data_number}_loss_{criterion}'
else:
    test_path = f'./test_result/test_{max_model_num}_model_{model_name}_epochs_{epochs}_data_{data_number}_loss_{criterion}'
os.makedirs(f'{test_path}')

if macro == 'on':
    chrome_button = {
        'step1': {'x': 1228, 'y': 214},
        'step2': {'x': 1794, 'y': 42},
        'step3': {'x': 857, 'y': 64},
        'step4': {'x': 869, 'y': 76}
    }

    x1 = chrome_button['step1']['x']
    y1 = chrome_button['step1']['y']
    x2 = chrome_button['step2']['x']
    y2 = chrome_button['step2']['y']

    # =======================================================
    # network order 입력
    m.moveTo(x1, y1, duration=2)
    m.click(clicks=3)

    time.sleep(1)
    m.press('delete')

    time.sleep(1)
    m.press('enter')

    time.sleep(1)
    m.click()

    time.sleep(1)
    m.typewrite(f'network_order = {network_order + 1}')

    # =======================================================
    # start
    m.moveTo(x2, y2, duration=2)
    m.click()