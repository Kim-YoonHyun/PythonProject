# normalize 할 때 각 data 마다의 최댓값을 사용하는 것이 아닌 전체 data에서의 최댓값을 사용하여 normalize
# variable scope를 설정

# 10_5
# macro 가능하게 설정

# 10_5_2
# input data 에서 edgefeature 대신 0 을 임의의 갯수 만큼 붙여서 함

# 10_5_3
# Macro 4 와 연동되게 함

# 10_5_4
# pd 파일 생성 추가

# _11
# Pytorch 에 적용된 설정 가져오기
# train, val, val 로 나누기 등
# learning rate 관련 추가
# _1 Macro_9 와 연동

# training (make_training_model_11 에서 이름 변경)
# word 설명과 맞추기
# Functions import 방식 변경
# 저장되는 data 중 skin point, score function type, pd 제거

import tensorflow as tf
import numpy as np
import time
import Networks.MLP_k_seq_parametric_networks.no_30_3 as network
# import Networks.zero_rev_ch_filter_parametric_networks.N12_0_256_256_256_f3 as network
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import Functions_mk2  # 오류 무시 가능
import math
import pyautogui as m


tf.set_random_seed(1)


# ############################## parameter #############################################################################
# A.
# data number for training
data_num = 101

# train, val, val
val_percentage = 20
test_percentage = 20

# data setting
labeling = 'on'

# Multi Targeting
Multi_Targeting = 'sub'

# loss 타입 입력 L1, L2
loss_type = 'L2'

# epochs
epochs = 1

# 학습 도중 val 를 입력할 epoch 스퀀스
val_epoch = 1

# learning rate 설정
starter_learning_rate = 0.1
decay_rate = 0.84

# on 일경우 train set 은 2로 고정, 그외의 경우 모든 train set 사용.
code_testing = 'on'

# Macro
macro = 'fon'
MT_order = 2
data_order = 5

# ############################## range setting #########################################################################
# model number
max_model_num = len(os.walk('./training_model/').__next__()[1]) + 1


# ############################## train/val set ########################################################################
# path 설정
if labeling == 'on':
    path = f'./data/data{data_num}(Label)'
else:
    path = f'./data/data{data_num}'

# load array
if Multi_Targeting == 'sub':
    data = np.load(path + f'/MT_data_sub.npy')
elif Multi_Targeting == 'concat':
    data = np.load(path + f'/MT_data_concat.npy')
elif Multi_Targeting == 'sub_concat':
    data = np.load(path + f'/MT_data_sub_concat.npy')
else:
    data = np.load(path + f'/data.npy')
GT_data = np.load(path + f'/score_add_dummy.npy')
num_skin_points = np.load(path + f'/num_skin_points.npy')

# variables
num_all_points = GT_data.shape[1]
num_all_data = GT_data.shape[0]
num_test_data = int(num_all_data * test_percentage/100)
num_val_data = int((num_all_data - num_test_data) * val_percentage/100)
num_train_data = num_all_data - num_val_data - num_test_data

# data 분류
train_data = []
train_GT_data = []
val_data = []
val_GT_data = []
test_data = []
test_GT_data = []

for i in range(num_all_data):
    if i < num_train_data:
        train_data.append(data[i])
        train_GT_data.append(GT_data[i])
    elif i >= num_train_data and i < num_train_data + num_val_data:
        val_data.append(data[i])
        val_GT_data.append(GT_data[i])
    else:
        test_data.append(data[i])
        test_GT_data.append(GT_data[i])

train_data = np.array(train_data)
train_GT_data = np.array(train_GT_data)
val_data = np.array(val_data)
val_GT_data = np.array(val_GT_data)
test_data = np.array(test_data)
test_GT_data = np.array(test_GT_data)

print(f'size of data   : {data.shape}')
print(f'size of GT data: {GT_data.shape}\n')
print(f'size of train data      : {train_data.shape}')
print(f'size of validation data : {val_data.shape}\n')
print(f'size of train GT data     : {train_GT_data.shape}')
print(f'size of validation GT data: {val_GT_data.shape}')


# ############################## Network ###############################################################################
# variable
size = data.shape

# placeholder
input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, size[1], size[2]))
gt_tensor = tf.placeholder(dtype=tf.float32, shape=(1, num_all_points))
is_training = tf.placeholder(dtype=tf.bool)


# layers
layer_names = []

# layer1 : EdgeConv layer
with tf.variable_scope('Training_model'):
    output_tensor = network.get_model(input_tensor, is_training)
network_name = os.path.basename(os.path.abspath(network.__file__)).split('.')[0].split('_')
network_name_len = len(network_name)
line = ''
for i in range(network_name_len):
    line += network_name[i]
    if i != (network_name_len - 1):
        line += '_'

layer1_name = line
layer_names.append(layer1_name)
layer_names = np.array(layer_names)

# loss
if loss_type == 'L1':
    L1_loss = output_tensor - gt_tensor
    loss_op = tf.reduce_mean(tf.abs(L1_loss))
elif loss_type == 'L2':
    L2_loss = (output_tensor - gt_tensor) ** 2
    loss_op = tf.reduce_mean(L2_loss)

# learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = starter_learning_rate
decay_steps = val_epoch * num_train_data
decay_rate = decay_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=global_step)

# Session
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.InteractiveSession(config=config)
with sess.as_default():
    sess.run(tf.global_variables_initializer())


# ############################## Training ##############################################################################
if __name__ == "__main__":  # 파이썬 인터프리터에 의해서 직접 실행될 경우
    # variable
    all_train_aver_loss = []
    all_train_aver_accuracy = []
    all_val_aver_loss = []
    all_val_aver_accuracy = []
    all_val_aver_accuracy_floor = []

    # training
    start = time.time()

    norm_train_data = Functions_mk2.normalize_data(train_data, num_skin_points, num_all_points)
    norm_train_GT_data = Functions_mk2.normalize_data(train_GT_data, num_skin_points, num_all_points)

    norm_val_data = Functions_mk2.normalize_data(val_data, num_skin_points, num_all_points)
    norm_val_GT_data = Functions_mk2.normalize_data(val_GT_data, num_skin_points, num_all_points)

    for epoch in range(epochs):
        epoch += 1
        train_loss_sum = 0
        for idx, data in enumerate(norm_train_data):
            if code_testing == 'on' and idx == 2:
                break
            [output, train_loss, _] = sess.run([output_tensor, loss_op, train_op],
                                         feed_dict={input_tensor: [data],
                                                    gt_tensor: [norm_train_GT_data[idx]],
                                                    is_training: True})

            train_loss_sum += train_loss

        train_aver_loss = (train_loss_sum/num_train_data) ** 0.5
        train_aver_accuracy = (1 - train_aver_loss) * 100

        print(f'epoch = %4d/{epochs}, lr = %.6f, loss = %.4f = %.2f' % (epoch, sess.run(optimizer._lr),
                                                                        train_aver_loss, train_aver_accuracy) + '%')

        all_train_aver_loss.append(train_aver_loss)
        all_train_aver_accuracy.append(train_aver_accuracy)

        # validation
        if epoch % val_epoch == 0:
            val_loss_sum = 0
            for jdx, data in enumerate(norm_val_data):
                if code_testing == 'on' and jdx == 2:
                    break
                val_output = sess.run(output_tensor, feed_dict={input_tensor: [data], is_training: True})

                # val loss
                val_loss = np.average(np.square(np.subtract(val_output[0], norm_val_GT_data[jdx])))
                val_loss_sum += val_loss

            val_aver_loss = (val_loss_sum/num_val_data) ** 0.5
            val_aver_accuracy = (1 - val_aver_loss) * 100

            all_val_aver_loss.append(val_aver_loss)
            all_val_aver_accuracy.append(val_aver_accuracy)
            all_val_aver_accuracy_floor.append(math.floor(val_aver_accuracy))

            print(f'val accuracy = {val_aver_accuracy}% ')
            print(Functions.make_percentage_bar(val_aver_accuracy, 100) + '\n')
            plt.plot(epoch, train_aver_loss, 'b.')
            plt.plot(epoch, val_aver_loss, 'r.')

        # train loss 및 val loss epoch graph 기록

        plt.plot(epoch, train_aver_loss, 'b.')
        plt.pause(0.1)
        train_aver_loss = 0

    # epoch graph 저장
    plt.plot(epoch, train_aver_loss, label='train loss')
    plt.plot(epoch, val_aver_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('epochs graph')


# ############################## 각종 정보 기록 #########################################################################
    if max_model_num < 10:
        train_path = f'./training_model/0{max_model_num}_model_{loss_type}_{data_num}_{epochs}'
        test_path = f'./test_result/test_0{max_model_num}_model_{loss_type}_{data_num}_{epochs}'
    else:
        train_path = f'./training_model/{max_model_num}_model_{loss_type}_{data_num}_{epochs}'
        test_path = f'./test_result/test_{max_model_num}_model_{loss_type}_{data_num}_{epochs}'
    order = 1

    # 저장할 파일 만들기
    os.makedirs(f'{train_path}')

    # 학습 모델 저장
    saver_path = saver.save(sess, f'{train_path}/model.ckpt')
    # tf.io.write_graph(sess.graph_def, '.', f'{train_path}/graph.pd', as_text=False)
    # tf.io.write_graph(sess.graph_def, '.', f'{train_path}/graph.pdtxt', as_text=True)
    # tf.train.write_graph(sess.graph.as_graph_def(), f"{train_path}", "graph.pdtxt")
    # tf.train.write_graph(sess.graph.as_graph_def(), f"{train_path}", "graph.pdtxt")

    # =========================
    # training 시간 기록
    training_time, hour, minute, sec = Functions.record_time(start)
    print('\n')
    print('run time :', training_time, 'sec')
    print('\n')

    # =========================
    # epoch graph 저장
    fig = plt.gcf()  # 변경한 곳
    fig.savefig(f'{train_path}/{order}; epoch gragh.png')
    plt.close(fig= fig)
    order += 1

    # =========================
    # 학습 정보 & 데이터 저장
    file_train_info = open(f'{train_path}/{order}; Training information.txt', 'w', encoding= 'utf8')
    line = f'< model_{loss_type}_{data_num}_{epochs} >\n\n' \
        f'Loss type : {loss_type}_norm\n' \
        f'data number : {data_num}\n' \
        f'equalizing type : {Multi_Targeting}\n' \
        f'epochs : {epochs}\n' \
        f'number of train data : {num_train_data}\n' \
        f'Time : {training_time} sec = {hour}시간 {minute}분 {sec}초\n\n' \
        f'Max val accuracy : {round(max(all_val_aver_accuracy), 2)}%, epoch {(all_val_aver_accuracy.index(max(all_val_aver_accuracy))+1)*val_epoch}\n' \
        f'Max val accuracy(1% 단위) : {max(all_val_aver_accuracy_floor)}%, epoch {(all_val_aver_accuracy_floor.index(max(all_val_aver_accuracy_floor))+1)*val_epoch}\n\n' \
        f'<network>\n'
    file_train_info.write(line)
    for i in range(layer_names.shape[0]):
        file_train_info.write(f'layer {i+1} : {layer_names[i]}\n')
    file_train_info.write('\n\n')

    for i in range(epochs):
        file_train_info.write(f'Training loss after {i+1} epoch = %.4f = %.2f' %(all_train_aver_loss[i], all_train_aver_accuracy[i]) + '%\n')
        if (i+1) % val_epoch == 0:
            file_train_info.write(f'\nval accuracy at {i+1} epoch = %.4f = %.2f ' %(all_val_aver_loss[i // val_epoch], all_val_aver_accuracy[i // val_epoch]) + '%\n\n')
    file_train_info.close()
    np.save(f'{train_path}/test_data', test_data)
    np.save(f'{train_path}/test_GT_data', test_GT_data)
    np.save(f'{train_path}/data_number', data_num)
    # val 결과 저장용 파일 저장
    os.makedirs(f'{test_path}')

    if macro == 'on':
        chrome_button = {
            'step1': {'x': 2451, 'y': 409},
            'step2': {'x': 2414, 'y': 499},
            'step3': {'x': 3649, 'y': 65}
        }

        x1 = chrome_button['step1']['x']
        y1 = chrome_button['step1']['y']
        x2 = chrome_button['step2']['x']
        y2 = chrome_button['step2']['y']
        x3 = chrome_button['step3']['x']
        y3 = chrome_button['step3']['y']

        # =======================================================
        # type MT order
        m.moveTo(x1, y1, duration=2)
        m.click(clicks=3)

        time.sleep(1)
        m.press('delete')

        time.sleep(1)
        m.press('enter')

        time.sleep(1)
        m.click()

        time.sleep(1)
        if MT_order != 2:
            m.typewrite(f'MT_order = {MT_order + 1}')
        else:
            m.typewrite(f'MT_order = 0')

        # =======================================================
        # type model number
        m.moveTo(x2, y2, duration=2)
        m.click(clicks=3)

        time.sleep(1)
        m.press('delete')

        time.sleep(1)
        m.press('enter')

        time.sleep(1)
        m.click()

        time.sleep(1)
        if MT_order != 2:
            m.typewrite(f'data_order = {data_order}')
        else:
            m.typewrite(f'data_order = {data_order + 1}')

        # =======================================================
        # start
        m.moveTo(x3, y3, duration=2)
        m.click(clicks=1)

    exit()
