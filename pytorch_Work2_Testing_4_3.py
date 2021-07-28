# _2_1
# shuffled 추가, title 길이 자동화

# _3
# loss 를 직접계산하는 것이 아닌 nn.MSE 를 사용, dummy 사용 여부 추가
# _3_1 model load 하는 과정에서 더 많은 정보를 불러옴

# _4
# dummy 관련 전부 다 삭제
# _4_1
# Macro_8 과 연동
# _4_2
# 최적점을 0 에서 1 로 수정
# _4_3
# 최적점 찾아내는 방식 변경

import torch
import torch.nn as nn
import numpy as np
import os
import Functions
import Functions_mk2
import vtk
import time
import pyautogui as m
# torch.manual_seed(289343610626200)


use_cuda = torch.cuda.is_available()
print('GPU 사용 가능 여부:', use_cuda)

device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

print('cuda index:', torch.cuda.current_device())
print('gpu 개수:', torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())
cuda = torch.device('cuda')
print(cuda)

print('cuda seed: ', torch.cuda.initial_seed())


# ######################################## parameter ###################################################################
# model_number
model_number = 54

# Min or Max
switch = 'Min'

# test set percentage (백분율 기준)
test_percentage = 20

# macro 사용 여부
macro = 'fon'

# Rendering camera 관련 (find_rendering_carmera_2.py 코드에서 추출)
pos_x = 15.422777069164356
pos_y = 381.6443091847143
pos_z = -58.814906569805544
foc_x = 10.503540519948022
foc_y = -35.84464739674654
foc_z = 20.611816652768752
view_x = -0.0003902626983822925
view_y = -0.18689197540848262
view_z = -0.9823803933420808
clip_x = 417.9918285014825
clip_y = 583.8280146395812
size_x = 1312
size_y = 577

position = 130

# title setting
title_row = 3
title_blank = 4

# 계산 관련
k_value = 10
Skin_label = 1

# point vertex size
model_point_size = 10
pmv_point_size = 20
tkav_point_size = 20
gt_point_size = 10
gt_entry_point_size = 20

# point vertex color
# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 point_actor 함수에 원하는 색깔 추가 가능
pmv_point_color = 'Gray'
tkav_point_color = 'Dark Gray'
gt_entry_point_color = 'Black'


# ######################################## path setting ################################################################
model_list = os.listdir(f'./trained_model')
name_length_list = []
for i in range(len(model_list)):
    name_length_list.append(len(model_list[i]))
string_length = max(name_length_list)

Functions.model_list_title(file_path=f'./trained_model', row_num=title_row, string_length=string_length, blank_num=title_blank)
print(f'Enter the model number for testing (1 ~ {len(model_list)})')
model_num = int(input())
model_name = model_list[model_num - 1]
print(model_name)
print('testing...\n\n')

train_path = f'./trained_model/{model_name}'
test_path = f'./test_result/test_{model_name}'


# ######################################## range setting ###############################################################
# path
data_number = np.load(f'{train_path}/data_number.npy')
data_path = f'./data/data(Label)_{data_number}'

# load array
all_data = np.load(data_path + f'/data.npy')
num_skin_points = np.load(f'{data_path}/num_skin_points.npy')
# entry = np.load(data_path + f'/entry.npy')
gt_data = np.load(data_path + f'/score_add_dummy.npy')

# variables
all_data_num = all_data.shape[0]
all_points_num = all_data.shape[1]
test_data_num = round(all_data_num * (test_percentage/100))

# test set separate
test_data = all_data[all_data_num - test_data_num:, :, :]

# gt test set
test_gt_data = gt_data[all_data_num - test_data_num:]

# test entry
# test_entry = entry[all_data_num - test_data_num:, :]

# print
print('size of all data       : ', all_data.shape)
print('size of gt data (score): ', gt_data.shape)
# print('size of entry          : ', entry.shape)

print('size of test data: ', test_data.shape)
print('size of test gt data: ', test_gt_data.shape)
# print('size of test entry: ', test_entry.shape)

# normalize
norm_test_data = Functions_mk2.normalize_data(test_data, num_skin_points, all_points_num)
norm_test_gt_data = Functions_mk2.normalize_data(test_gt_data, num_skin_points, all_points_num)


# ######################################## Rendering ###################################################################
# renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# render window
renWin = vtk.vtkRenderWindow()
renWin.SetSize(size_x, size_y)
renWin.AddRenderer(ren)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(interactorStyle)

# render camera setting
camera = vtk.vtkCamera()
camera.SetPosition(pos_x, pos_y, pos_z)
camera.SetFocalPoint(foc_x, foc_y, foc_z)
camera.SetViewUp(view_x, view_y, view_z)
camera.SetClippingRange(clip_x, clip_y)
ren.SetActiveCamera(camera)


# ######################################## network #####################################################################
criterion = nn.MSELoss()
model = torch.load(f'{train_path}/trained_model.pth')
model.load_state_dict(torch.load(f'{train_path}/model_state_dict.pth'))
checkpoint = torch.load(f'{train_path}/all.tar')
model.load_state_dict(checkpoint['net'])
# model.eval()


# ######################################## test ########################################################################
# numpy 를 torch 로 변환
norm_test_data = torch.from_numpy(norm_test_data).float()
norm_test_gt_data = torch.from_numpy(norm_test_gt_data).float()

flag = 5

if switch == 'Min':
    optimal_score = 0.0
elif switch == 'Max':
    optimal_score = 1.0
else:
    optimal_score = 1000

testing_time_list = []
k_value = k_value + 1
test_index = 0

while True:
    print('enter the test index')
    test_index = int(input())
    if test_index == 10000:
        exit()
    start = time.time()

# while test_index < test_data_num:
#     start = time.time()
    input_data = torch.unsqueeze(torch.unsqueeze(torch.transpose(norm_test_data[test_index], 0, 1), 0), 2).float()
    # input_data = norm_test_data[test_index]

    if use_cuda:
        input_data = input_data.cuda()

    # testing
    test_output = model(input_data)

    # GPU 세팅
    if device.type == 'cuda':
        print('Memory Usage')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    test_output = test_output.cpu()
    test_loss = criterion(test_output, norm_test_gt_data[test_index])
    loss = test_loss.item()**0.5

    # numpy 로 변경
    test_output = test_output.detach().numpy()[:num_skin_points]
    norm_test_gt_data_skin = norm_test_gt_data[test_index][0:num_skin_points]

    # loss 계산
    if flag == 5:
        norm_test_gt_data = norm_test_gt_data.numpy()
        flag = 0

    # 랜더링용 surface
    test_surface = test_data[test_index][0:num_skin_points, 0:3]
    optimal_point_index = np.array(np.where(norm_test_gt_data_skin == optimal_score))[0][0]

    # min vertex 계산
    gt_score_min_vertex = test_surface[optimal_point_index]
    if switch == 'Min':
        predicted_min_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(test_output, test_data[:, 0:num_skin_points, 0:3][test_index], 1,
                                                                       1, 1)
        top_k_aver_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(test_output, test_data[:, 0:num_skin_points, 0:3][test_index],
                                                                    k_value, 1, 1)
    elif switch == 'Max':
        predicted_min_vertex = Functions_mk2.make_top_k_average_vertex_mk2(test_output,
                                                                           test_data[:, 0:num_skin_points, 0:3][test_index],
                                                                           1, 1, 1)
        top_k_aver_vertex = Functions_mk2.make_top_k_average_vertex_mk2(test_output,
                                                                        test_data[:, 0:num_skin_points, 0:3][test_index],
                                                                        k_value, 1, 1)

    # calculate D
    pmv_D = Functions.euclidean_distance(predicted_min_vertex, gt_score_min_vertex)
    tkav_D = Functions.euclidean_distance(top_k_aver_vertex, gt_score_min_vertex)
    all_D = np.array([pmv_D, tkav_D])



    # for i in range(2500):
    #     print(norm_test_gt_data[test_index][i], i)
    # print(min(norm_test_gt_data[test_index]))
    #
    # print(a)
    # print(test_surface[2385])
    # print(test_entry[test_index])
    # exit()

    model_actor = Functions.point_actor(test_surface, model_point_size, 'None')
    Functions.assign_value_to_polydata(model_actor.GetMapper().GetInput(), test_output[:num_skin_points])
    model_actor.SetPosition(-position, 0, 0)

    # predicted entry actor
    model_pmv_actor = Functions.point_actor(predicted_min_vertex, pmv_point_size, pmv_point_color)
    model_tkav_actor = Functions.point_actor(top_k_aver_vertex, tkav_point_size, tkav_point_color)

    model_pmv_actor.SetPosition(-position, 0, 0)
    model_tkav_actor.SetPosition(-position, 0, 0)

    # gt_actor
    gt_actor = Functions.point_actor(test_surface, gt_point_size, 'None')
    Functions.assign_value_to_polydata(gt_actor.GetMapper().GetInput(), norm_test_gt_data[test_index][0:num_skin_points])
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

    # save rendering image
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"{test_path}/rendering of {test_index + 1}.png")
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    # renderWindow.RemoveRenderer(renderer)
    ren.RemoveActor(model_actor)
    ren.RemoveActor(model_pmv_actor)
    ren.RemoveActor(model_tkav_actor)
    ren.RemoveActor(gt_actor)
    ren.RemoveActor(gt_entry_actor1)
    ren.RemoveActor(gt_entry_actor2)

    # accuracy
    accuracy = (1 - loss) * 100

    # percentage bar
    bar = Functions.make_percentage_bar(round((1 - loss) * 100, 1), 50)

    # testing time
    testing_time, _, _, _ = Functions.record_time(start)
    testing_time_list.append(testing_time)

    # print
    print(f'%2d/%2d  {bar}  test accuracy: %.2f' % (test_index + 1, test_data_num, accuracy) + " %")
    print('run time :', testing_time, 'sec\n')


# ################################################## 결과 저장 #########################################################
    try:
        if not (os.path.isdir(f'{test_path}/D & loss')):
            os.makedirs(f'{test_path}/D & loss')
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise

    testindex_k_testloss = []
    testindex_k_testloss.append(test_index)
    testindex_k_testloss.append(k_value)
    testindex_k_testloss.append(loss)
    testindex_k_testloss = np.array(testindex_k_testloss)

    np.save(f'{test_path}/D & loss/{test_index + 1}_k_test_loss', testindex_k_testloss)
    np.save(f'{test_path}/D & loss/{test_index + 1}_all_D', all_D)
    np.save(f'{test_path}/D & loss/{test_index + 1}_test_output', test_output)

    test_index += 1

    # loss 계산
    if flag == 0:
        norm_test_gt_data = torch.from_numpy(norm_test_gt_data)
        flag = 5

# =======================================================
layer_names = []

# layer1 : EdgeConv layer
# network_name = os.path.basename(os.path.abspath(network.__file__)).split('.')[0].split('_')
# network_name_len = len(network_name)
# line = ''
# for i in range(network_name_len):
#     line += network_name[i]
#     if i != (network_name_len - 1):
#         line += '_'

# network_name2 = line
# network_txt = open(f'{test_path}/{network_name2}.txt', 'w', encoding= 'utf8')
# network_txt.write(f'{network_name2}')
# network_txt.close()

# testing time 저장
testing_time_array = np.array(testing_time_list)
np.save(f'{test_path}/testing_time_array', testing_time_array)


# ##################################################### Macro ######################################################

if macro == 'on':
    # step 에 따른 마우스 위치
    chrome_button = {
        'step1': {'x': 1303, 'y': 216},
        'step2': {'x': 1796, 'y': 45},
        'step3': {'x': 271, 'y': 964},
        'step4': {'x': 869, 'y': 76}
    }

    x1 = chrome_button['step1']['x']
    y1 = chrome_button['step1']['y']
    x2 = chrome_button['step2']['x']
    y2 = chrome_button['step2']['y']
    x3 = chrome_button['step3']['x']
    y3 = chrome_button['step3']['y']

    # =======================================================
    # type model number
    m.moveTo(x1, y1, duration=2)
    m.click(clicks=3)

    time.sleep(1)
    m.press('delete')

    time.sleep(1)
    m.press('enter')

    time.sleep(1)
    m.click()

    time.sleep(1)
    m.typewrite(f'model_number = {model_number + 1}')

    # =======================================================
    # start
    m.moveTo(x2, y2, duration=2)
    m.click(clicks=1)

exit()