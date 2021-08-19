# sampling 된 data 를 확인하기 위한 랜더링 코드
# score 는 적용되어 있지 않으며 sampling 형상을 확인하기 위함
# (label) 폴더를 구분하는 방식 추가
# data 31번 부터 적용가능

# _0
# work_see_GT_data.py 와 work_see_sampling_data_3 코드를 합치고 선택지를 구성한 코드

# _1
# code renewal


import numpy as np
import sys
import vtk
import os
import dill
import functions_my
import vtk
import functions_vtk

# <Rendering camera 관련>-----------------------------------------------
position = 20

# render 카메리 상세 설정시(work_find_rendering_camera 에서 수치 추출 가능)
size_x = 250
size_y = 250
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
target_point_size = 5
bone1_point_size = 5
bone2_point_size = 5
skin_point_size = 5

# point vertex color
# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
target_point_color = 'Black'
bone1_point_color = 'Dark Gray'
bone2_point_color = 'Gray'
skin_point_color = 'Pink'


# <vtk 기본 세팅>-----------------------------------------------------------
##### vtk 기본 세팅
# renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(int(size_x*3/2), int(size_y*3/2))

# Rendering 카메라 위치 설정
camera = vtk.vtkCamera()
camera.SetClippingRange(clip_x, clip_y)
camera.SetDistance(Dist)
camera.SetFocalPoint(foc_x, foc_y, foc_z)
camera.SetPosition(pos_x, pos_y, pos_z)
camera.SetThickness(Thick)
camera.SetViewUp(VU_x, VU_y, VU_z)
ren.SetActiveCamera(camera)

# Interactor 설정
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)

# <데이터 불러오기>------------------------------------------
functions_my.data_list_title(f'./data')
data_num = int(input('rendering 할 데이터 번호 선택 > '))
data_path = f'./data/data{data_num}'
with open(f'{data_path}/data_information.pkl', 'rb') as file:
    target_data, bone1_data, bone2_data, skin_data = dill.load(file)

for i in range(target_data.num_of_data):

    # target, bone actor
    target_actor = functions_vtk.point_actor(target_data.data_vertices_list[i], target_point_size, target_point_color)
    target_actor.SetPosition(0, 0, 0)

    bone1_actor = functions_vtk.point_actor(bone1_data.data_vertices_list[i], bone1_point_size, bone1_point_color)
    bone1_actor.SetPosition(0, 0, 0)

    bone2_actor = functions_vtk.point_actor(bone2_data.data_vertices_list[i], bone2_point_size, bone2_point_color)
    bone2_actor.SetPosition(0, 0, 0)

    skin_actor = functions_vtk.point_actor(skin_data.data_vertices_list[i], skin_point_size, skin_point_color)
    skin_actor.SetPosition(0, 0, 0)


    # 아직 구현 불가능
    # skin_actor = functions_vtk.point_actor(skin_data.data_vertices_list, skin_point_size, 'None')
    # functions_vtk.assign_value_to_polydata(skin_actor.GetMapper().GetInput(), [0, 0.5, 1])
    # model_actor_distribution.SetPosition(-position, 0, 0)

    # add actor
    ren.AddActor(target_actor)
    ren.AddActor(bone1_actor)
    ren.AddActor(bone2_actor)
    ren.AddActor(skin_actor)

    # renderWindow initialize
    renWin.Render()
    iren.Initialize()

    # 랜더링 이미지 저장
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"{data_path}/data_image/index{i}.png")
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    iren.Start()

    # 반복 렌더링이 필요한 경우
    # renderWindow.RemoveRenderer(renderer)
    ren.RemoveActor(target_actor)
    ren.RemoveActor(bone1_actor)
    ren.RemoveActor(bone2_actor)
    ren.RemoveActor(skin_actor)









exit()

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