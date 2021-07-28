# sampling 된 data 를 확인하기 위한 랜더링 코드
# score 는 적용되어 있지 않으며 sampling 형상을 확인하기 위함
# (label) 폴더를 구분하는 방식 추가
# data 31번 부터 적용가능

# _0
# work_see_GT_data.py 와 work_see_sampling_data_3 코드를 합치고 선택지를 구성한 코드

import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import vtk
import os


# ############################## parameter #############################################################################
# A.
# on or else. labeling 사용 유무 결정.
use_labeling = 'on'

# point size 결정
Skin_point_size = 5
Bone1_point_size = 5
Bone2_point_size = 5
target_point_size = 5
gt_point_size = 10

# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
Skin_point_color = 'Pink'
Bone1_point_color = 'Dark Gray'
Bone2_point_color = 'Gray'
target_point_color = 'Yellow'
entry_point_color = 'Black'

# Add actor
add_gt_actor = 'on'
add_B2_actor = 'on'
add_B1_actor = 'on'
add_Skin_actor = 'on'
add_tar_actor = 'on'

# camera setting
pos_x = -18.585473656914566
pos_y = -657.2446462631391
pos_z = 6.053054945030928
foc_x = 10.503540519948022
foc_y = -35.84464739674654
foc_z = 20.611816652768752
view_x = -0.06724420983445002
view_y = 0.026515522978191545
view_z = -0.9973841503076605
clip_x = 444.06450387938065
clip_y = 825.8727701044749
size_x = 922
size_y = 743


# ############################## data range setting ####################################################################
max_set_num = len(os.walk('./stl/').__next__()[1])
max_data_num = len(os.walk('./data/').__next__()[1])

set_num = []
for i in range(1, max_set_num + 1):
    set_num.append(i)
set_num = np.array(set_num)
target_property_num = 4


# ############################## read data file ########################################################################
print('data 31번 부터 적용되는 코드')
print(f'Enter the data number to use (31 ~ {max_data_num})')
data_num = int(input())

if use_labeling == 'on':
    path = f'./data/data{data_num}(Label)'
else:
    path = f'./data/data{data_num}'

data = np.load(path + f'/data.npy')
score = np.load(path + f'/score.npy')
num_sur_points = np.load(path + f'/num_skin_points.npy')
num_B1_points = np.load(path + f'/num_bone1_points.npy')
num_B2_points = np.load(path + f'/num_bone2_points.npy')
num_tar_points = 1

print(f'data shape : {data.shape}\n\n')

# ==============================
# Renderer
ren = vtk.vtkRenderer()

# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(size_x, size_y)

# Render window interactor
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)

# =========================
# Rendering camera setting
camera = vtk.vtkCamera()
camera.SetPosition(pos_x, pos_y, pos_z)
camera.SetFocalPoint(foc_x, foc_y, foc_z)
camera.SetViewUp(view_x, view_y, view_z)
camera.SetClippingRange(clip_x, clip_y)

ren.SetActiveCamera(camera)

print(f'B. Enter the data type (1, 2)')
print(f'1: training data')
print(f'2: GT data')
data_type = int(input())
if data_type == 1:
    t = f'train'
elif data_type == 2:
    t = f'GT'

print('C. Enter the mode (see or saving)')
mode = str(input())

if mode == 'see':
    print(f'D. enter the sampling data number 0 ~ {data.shape[0] - 1}')
    index = int(input())
    print(f'D. press q in rendering window to see another data')
elif mode == 'saving':
    os.makedirs(f'./image/data{data_num}_{t}')
    index = 0
else:
    print('please type mode "saving" or "see"')
    index = None
    exit()

while index < data.shape[0]:
    # rendering 용 point cloud data
    skin = data[index][0:num_sur_points, 0:3]
    B1 = data[index][num_sur_points:num_sur_points + num_B1_points, 0:3]
    B2 = data[index][num_sur_points + num_B1_points:num_sur_points + num_B1_points + num_B2_points, 0:3]
    tar = data[index][num_sur_points + num_B1_points + num_B2_points:num_sur_points + num_B1_points + num_B2_points +
                                                                     num_tar_points, 0:3]

    # make actor
    B1_actor = Functions.point_actor(B1, Bone1_point_size, Bone1_point_color)
    B2_actor = Functions.point_actor(B2, Bone2_point_size, Bone2_point_color)
    tar_actor = Functions.point_actor(tar, target_point_size, target_point_color)
    Skin_actor = Functions.point_actor(skin, Skin_point_size, Skin_point_color)

    switch = np.load(f'{path}/score function type.npy')
    if switch == 'Max':
        find_score = max(score[index])
    elif switch == 'Min':
        find_score = min(score[index])
    optimal_vertex_index = np.array(np.where(score[index] == find_score))[0][0]
    entry = skin[optimal_vertex_index]

    normalize_GT = Functions.normalize_data(score)
    training_normalize_GT = normalize_GT[index]
    gt_actor = Functions.point_actor(skin, gt_point_size, 'None')
    Functions.assign_value_to_polydata(gt_actor.GetMapper().GetInput(), training_normalize_GT)

    # add actor
    if data_type == 1:
        if add_B1_actor == 'on':
            ren.AddActor(B1_actor)
        if add_B2_actor == 'on':
            ren.AddActor(B2_actor)
        if add_tar_actor == 'on':
            ren.AddActor(tar_actor)
        if add_Skin_actor == 'on':
            ren.AddActor(Skin_actor)
    elif data_type == 2:
        if add_B1_actor == 'on':
            ren.AddActor(B1_actor)
        if add_B2_actor == 'on':
            ren.AddActor(B2_actor)
        if add_tar_actor == 'on':
            ren.AddActor(tar_actor)
        if add_gt_actor == 'on':
            ren.AddActor(gt_actor)

    ren.SetBackground(1, 1, 1)
    renWin.Render()
    iren.Initialize()
    if mode == 'see':
        iren.Start()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    # 랜더링 이미지 저장
    if mode == 'saving':
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(f"./image/data{data_num}_{t}/rendering of {index}.png")
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    if data_type == 1:
        if add_B1_actor == 'on':
            ren.RemoveActor(B1_actor)
        if add_B2_actor == 'on':
            ren.RemoveActor(B2_actor)
        if add_Skin_actor == 'on':
            ren.RemoveActor(Skin_actor)
        if add_tar_actor == 'on':
            ren.RemoveActor(tar_actor)
    elif data_type == 2:
        if add_B1_actor == 'on':
            ren.RemoveActor(B1_actor)
        if add_B2_actor == 'on':
            ren.RemoveActor(B2_actor)
        if add_gt_actor == 'on':
            ren.RemoveActor(gt_actor)
        if add_tar_actor == 'on':
            ren.RemoveActor(tar_actor)

    if mode == 'see':
        for i in range(15):
            print('\n')
        print(f'D. enter the sampling data number 0 ~ {data.shape[0] - 1}')
        index = int(input())
        print(f'D. press q in rendering window to see another data')
    elif mode == 'saving':
        index = index + 1