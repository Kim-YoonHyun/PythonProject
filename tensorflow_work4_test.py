# training 할때와 다른 size를 data를 넣었을 때 작동 가능하도록 제작
# 모든 test set에 대해 반복을 통해 한번에 보는것이 불가능?
# scope 설정 떄문에 오류가 나기 때문에 이를 없앰.
# 하지만 scope 설정이 있고 없고 에 따른 이점을 아직 모름

# 위 문제 전부 해결
# 시간 늘어나는 문제 해결

# 15_6_2
# input data 에서 edge feature 대신 0 을 임의의 갯수 만큼 붙여서 함

# 15_6_3
# Macro 5와 연동되게설정

# 15_6_4
# network 이름 txt 추가 저장

# _16
# entry 직접계산

# test (score_predict_test_16 에서 이름 변경) >>> 20210618 지금까지 모든 결과를 추출한 code
# word 설명에 맞추기
# skin point 및 score type 은 데이터폴더에서 직접 불러옴

import tensorflow as tf
import numpy as np
import vtk
import Networks.MLP_k_seq_parametric_networks.no_30_3 as network

import os
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import Functions_mk2  # 오류 무시 가능
import time

# import pyautogui as m

tf.set_random_seed(1)

# ############################## parameter #############################################################################
# A.
# 계산 관련
k_value = 10

# Rendering camera 관련
position = 130

size_x = 1050
size_y = 535
clip_x = 437.48541978551657
clip_y = 832.9353965694179
Dist = 622.2508231247085
foc_x = -6.5353582725359445
foc_y = -34.446804088366704
foc_z = 15.888225067695487
pos_x = -28.334831438062928
pos_y = -653.3622452313335
pos_z = 76.42372344583254
Thick = 395.4499767839013
VU_x = 0.033045448043591226
VU_y = -0.09844414645719703
VU_z = -0.9945937604830992

# point vertex size
model_point_size = 10
pmv_point_size = 20

# point vertex color
# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
pmv_point_color = 'Gray'
tkav_point_color = 'Dark Gray'
gt_entry_point_color = 'Black'

# macro
macro_testing = 'ona'


# ############################## range setting #########################################################################
max_model_num = len(os.walk('./test_result/').__next__()[1])
k_value = k_value + 1


# ############################## Enter the parameter ###################################################################
print(f'B. Enter the model number to use for test (1 ~ {max_model_num})')
model_num = int(input())
model_name = os.listdir(f"./training_model")[model_num-1]
print(model_name)
train_path = f'./training_model/{model_name}'
test_path = f'./test_result/test_{model_name}'
data_number = np.load(f'{train_path}/data_number.npy')
data_path = f'./data/data{data_number}(Label)'




# ############################## Read model ############################################################################
test_data = np.load(f'{train_path}/test_data.npy')[1:]
test_GT_data = np.load(f'{train_path}/test_GT_data.npy')[1:]
num_skin_points = np.load(f'{data_path}/num_skin_points.npy')

num_test_data = test_data.shape[0]
num_all_points = test_data.shape[1]
size = test_data.shape

norm_test_data = Functions_mk2.normalize_data(test_data, num_skin_points, num_all_points)
norm_test_GT_data = Functions_mk2.normalize_data(test_GT_data, num_skin_points, num_all_points)


# ############################## Rendering #############################################################################
# renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(int(size_x*3/2), int(size_y*3/2))

# =========================
# Rendering 카메라 위치 설정
camera = vtk.vtkCamera()
camera.SetClippingRange(clip_x, clip_y)
camera.SetDistance(Dist)
camera.SetFocalPoint(foc_x, foc_y, foc_z)
camera.SetPosition(pos_x, pos_y, pos_z)
camera.SetThickness(Thick)
camera.SetViewUp(VU_x, VU_y, VU_z)
ren.SetActiveCamera(camera)

iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)


# ############################## network ###############################################################################
# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 탄력적으로 GPU memory 사용
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

# network 설정
input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, size[1], size[2]))
is_training = tf.placeholder(dtype=tf.bool)

with tf.variable_scope('Training_model', reuse=tf.AUTO_REUSE):
    output_tensor = network.get_model(input_tensor, is_training)
    # scope.reuse_variables() # 재사용에 필요한 코드

# 저장된 학습 모델 불러오기
saver = tf.train.Saver()
saver.restore(sess, f'{train_path}/model.ckpt')

# variables
test_index = 0
testing_times = []


# ############################## test ##################################################################################
# test set을 무한히 반복해서 넣는 경우
# while (True):
#     test_index = random.randrange(0, num_test_data)

# test 를 한번만 진행
switch = np.load(f'{data_path}/score function type.npy')
if switch == 'Min':
    optimal_score = 0.0
elif switch == 'Max':
    optimal_score = 1.0
else:
    optimal_score = 1000

num_test_data = test_data.shape[0] - 1
norm_test_data = norm_test_data[1:, :, :]
norm_test_GT_data = norm_test_GT_data[1:, :]

tkav_point_size = pmv_point_size
gt_point_size = model_point_size
gt_entry_point_size = pmv_point_size

while test_index < num_test_data:
    if test_index == 18 or test_index == 43:
        test_index += 1
        continue
    # 시간 기록
    start = time.time()

    # test 결과 계산
    test_output = sess.run(output_tensor, feed_dict={input_tensor: [norm_test_data[test_index]], is_training: True})
    test_loss = np.average(np.absolute(np.subtract(test_output[0], norm_test_GT_data[test_index])))

    # calculate entry vertices & D
    test_output = test_output[:, 0:num_skin_points]
    norm_test_GT_data_skin = norm_test_GT_data[test_index][0:num_skin_points]

    # surface
    test_surface = test_data[test_index][0:num_skin_points, 0:3]

    optimal_point_index = np.array(np.where(norm_test_GT_data_skin == optimal_score))[0][0]
    gt_score_min_vertex = test_surface[optimal_point_index]

    # calculate vertices
    if switch == 'Min':
        predicted_min_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(test_output,
                                                                           test_data[:, 0:num_skin_points, 0:3][
                                                                               test_index], 1,
                                                                           1, 1)
        top_k_aver_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(test_output,
                                                                        test_data[:, 0:num_skin_points, 0:3][
                                                                            test_index],
                                                                        k_value, 1, 1)
    elif switch == 'Max':
        predicted_min_vertex = Functions_mk2.make_top_k_average_vertex_mk2(test_output,
                                                                           test_data[:, 0:num_skin_points, 0:3][
                                                                               test_index],
                                                                           1, 1, 1)
        top_k_aver_vertex = Functions_mk2.make_top_k_average_vertex_mk2(test_output,
                                                                        test_data[:, 0:num_skin_points, 0:3][
                                                                            test_index],
                                                                        k_value, 1, 1)

    # calculate D
    pmv_D = Functions.euclidean_distance(predicted_min_vertex, gt_score_min_vertex)
    tkav_D = Functions.euclidean_distance(top_k_aver_vertex, gt_score_min_vertex)
    all_D = np.array([pmv_D, tkav_D])

    # model actor
    model_actor = Functions.point_actor(test_surface, model_point_size, 'None')
    Functions.assign_value_to_polydata(model_actor.GetMapper().GetInput(), test_output[0])
    model_actor.SetPosition(-position, 0, 0)

    # predicted entry actor
    model_pmv_actor = Functions.point_actor(predicted_min_vertex, pmv_point_size, pmv_point_color)
    model_tkav_actor = Functions.point_actor(top_k_aver_vertex, tkav_point_size, tkav_point_color)

    model_pmv_actor.SetPosition(-position, 0, 0)
    model_tkav_actor.SetPosition(-position, 0, 0)

    # gt_actor
    gt_actor = Functions.point_actor(test_surface, gt_point_size, 'None')
    Functions.assign_value_to_polydata(gt_actor.GetMapper().GetInput(), norm_test_GT_data_skin)
    gt_actor.SetPosition(position, 0, 0)

    # gt entry actor
    gt_entry_actor1 = Functions.point_actor([gt_score_min_vertex], gt_entry_point_size, gt_entry_point_color)
    gt_entry_actor1.SetPosition(position, 0, 0)
    gt_entry_actor2 = Functions.point_actor([gt_score_min_vertex], gt_entry_point_size, gt_entry_point_color)
    gt_entry_actor2.SetPosition(-position, 0, 0)

    # add actor
    ren.AddActor(model_actor)
    ren.AddActor(model_pmv_actor)
    ren.AddActor(model_tkav_actor)
    ren.AddActor(gt_actor)
    ren.AddActor(gt_entry_actor1)
    ren.AddActor(gt_entry_actor2)

    # renderWindow initialize
    renWin.Render()
    iren.Initialize()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    # 랜더링 이미지 저장
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"{test_path}/rendering of {test_index}.png")
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    # renderWindow.RemoveRenderer(renderer)
    ren.RemoveActor(model_actor)
    ren.RemoveActor(model_pmv_actor)
    ren.RemoveActor(model_tkav_actor)
    ren.RemoveActor(gt_actor)
    ren.RemoveActor(gt_entry_actor1)
    ren.RemoveActor(gt_entry_actor2)

    # ############################## 기록 ##############################################################################
    accuracy = round((1 - test_loss) * 100, 2)
    bar = Functions.make_percentage_bar(round((1 - test_loss) * 100, 1), 50)
    testing_time,_ ,_ ,_ = Functions.record_time(start)
    testing_times.append(testing_time)

    print(test_index)
    print(f'accuracy:  {accuracy}%  {bar}')
    print('run time :', testing_time, 'sec')

    # ############################## 결과 저장 #########################################################################
    try:
        if not(os.path.isdir(f'{test_path}/D & loss')):
            os.makedirs(f'{test_path}/D & loss')
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise

    testindex_k_testloss = []
    testindex_k_testloss.append(test_index)
    testindex_k_testloss.append(k_value)
    testindex_k_testloss.append(test_loss)
    testindex_k_testloss = np.array(testindex_k_testloss)

    np.save(f'{test_path}/D & loss/{test_index}_k_test_loss', testindex_k_testloss)
    np.save(f'{test_path}/D & loss/{test_index}_all_D', all_D)
    np.save(f'{test_path}/D & loss/{test_index}_test_output', test_output)

    test_index += 1

# =======================================================
layer_names = []

# layer1 : EdgeConv layer
network_name = os.path.basename(os.path.abspath(network.__file__)).split('.')[0].split('_')
network_name_len = len(network_name)
line = ''
for i in range(network_name_len):
    line += network_name[i]
    if i != (network_name_len - 1):
        line += '_'

network_name2 = line
network_txt = open(f'{test_path}/{network_name2}.txt', 'w', encoding='utf8')
network_txt.write(f'{network_name2}')
network_txt.close()

# testing time 저장
testing_times = np.array(testing_times)
np.save(f'{test_path}/testing_times', testing_times)

if macro_testing == 'on':
    chrome_button = {
        'step1': {'x': 1509, 'y': 253},
        'step2': {'x': 1765, 'y': 61},
        'step3': {'x': 1315, 'y': 690}
    }

    x1 = chrome_button['step1']['x']
    y1 = chrome_button['step1']['y']
    x2 = chrome_button['step2']['x']
    y2 = chrome_button['step2']['y']

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

    m.moveTo(x2, y2, duration=2)
    m.click()

exit()