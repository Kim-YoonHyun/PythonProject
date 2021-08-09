# sampling 된 data 를 확인하기 위한 랜더링 코드
# score 는 적용되어 있지 않으며 sampling 형상을 확인하기 위함
# (label) 폴더를 구분하는 방식 추가
# data 31번 부터 적용가능

# _3
# data 번호는 0으로 통일, camera setting 정보만 추출

# _4 
# 특정데이터 하나만 불러오는 방식으로 변경

import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\Me\\PycharmProjects\\My_Functions')
import Functions  # 오류 무시 가능
import vtk


# ############################## parameter #############################################################################
# A.
# Add actor 여부
add_B2_actor = 'on'
add_B1_actor = 'on'
add_Skin_actor = 'on'
add_tar_actor = 'on'

# on or else. labeling 사용 유무 결정.
use_labeling = 'on'
Skin_point_size = 10
Bone1_point_size = 10
Bone2_point_size = 10
target_point_size = 10

# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
Skin_point_color1 = 'Pink'
Skin_point_color2 = 'Light Red'
Bone1_point_color = 'Dark Gray'
Bone2_point_color = 'Gray'
target_point_color = 'Yellow'

# 랜더링 카메라 관련 변수들
size_x = 500
size_y = 500
clip_x = 189.7729726410838
clip_y = 701.0019797180688
Dist = 425.0056848061751
foc_x = 10.503540519948022
foc_y = -35.84464739674654
foc_z = 20.611816652768752
pos_x = -52.412492415378495
pos_y = -453.24475631615553
pos_z = 70.09468228636257
Thick = 511.229007076985
VU_x = 0.09812314339076937
VU_y = -0.1317303689489539
VU_z = -0.9864172335415108

position = 130


# ############################## read data file ########################################################################
path = f'./data_for_find_rendering_camera'

data = np.load(path + f'/data.npy')
num_sur_points = np.load(path + f'/number_of_Skin.npy')
num_B1_points = np.load(path + f'/number_of_Bone1.npy')
num_B2_points = np.load(path + f'/number_of_Bone2.npy')
num_tar_points = 1

# =======================================================
# Renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(size_x, size_y)

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

# Render window interactor
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)

index = 0
# rendering 용 point cloud data
sur = Functions.pick_up_array(case=data, case_number=index, range1=0, range2=num_sur_points)
while sur.shape[1] > 3:
    sur = np.delete(sur, (sur.shape[1] - 1), axis=1)

B1 = data[index][num_sur_points:num_sur_points + num_B1_points, 0:3]
B2 = data[index][num_sur_points + num_B1_points:num_sur_points + num_B1_points + num_B2_points, 0:3]
tar = data[index][num_sur_points + num_B1_points + num_B2_points:num_sur_points + num_B1_points + num_B2_points +
                                                                 num_tar_points, 0:3]

while index != 10000:
    ren.ResetCamera()
    print(' \n──  Press q in render window to see camera setting ── ')

    # =======================================================
    # make actor
    B1_actor = Functions.point_actor(B1, Bone1_point_size, Bone1_point_color)
    B2_actor = Functions.point_actor(B2, Bone2_point_size, Bone2_point_color)
    tar_actor = Functions.point_actor(tar, target_point_size, target_point_color)
    Skin_actor1 = Functions.point_actor(sur, Skin_point_size, Skin_point_color1)
    Skin_actor2 = Functions.point_actor(sur, Skin_point_size, Skin_point_color2)
    Skin_actor3 = Functions.point_actor(sur, Skin_point_size, Skin_point_color2)

    Skin_actor1.SetPosition(-position, 0, 0)
    Skin_actor2.SetPosition(position, 0, 0)
    Skin_actor3.SetPosition(0, 0, position)

    # add actor
    if add_B1_actor == 'on':
        ren.AddActor(B1_actor)
    if add_B2_actor == 'on':
        ren.AddActor(B2_actor)
    if add_tar_actor == 'on':
        ren.AddActor(tar_actor)
    if add_Skin_actor == 'on':
        ren.AddActor(Skin_actor1)
        ren.AddActor(Skin_actor2)
        ren.AddActor(Skin_actor3)

    ren.SetBackground(1, 1, 1)

    renWin.Render()
    iren.Start()

    for i in range(15):
        print('\n')
    print(f'size_x = {renWin.GetSize()[0]}')
    print(f'size_y = {renWin.GetSize()[1]}')
    print(f'clip_x = {camera.GetClippingRange()[0]}')
    print(f'clip_y = {camera.GetClippingRange()[1]}')
    print(f'Dist = {camera.GetDistance()}')
    print(f'foc_x = {camera.GetFocalPoint()[0]}')
    print(f'foc_y = {camera.GetFocalPoint()[1]}')
    print(f'foc_z = {camera.GetFocalPoint()[2]}')
    print(f'pos_x = {camera.GetPosition()[0]}')
    print(f'pos_y = {camera.GetPosition()[1]}')
    print(f'pos_z = {camera.GetPosition()[2]}')
    print(f'Thick = {camera.GetThickness()}')
    print(f'VU_x = {camera.GetViewUp()[0]}')
    print(f'VU_y = {camera.GetViewUp()[1]}')
    print(f'VU_z = {camera.GetViewUp()[2]}')

    if add_B1_actor == 'on':
        ren.RemoveActor(B1_actor)
    if add_B2_actor == 'on':
        ren.RemoveActor(B2_actor)
    if add_Skin_actor == 'on':
        ren.RemoveActor(Skin_actor1)
        ren.RemoveActor(Skin_actor2)
        ren.RemoveActor(Skin_actor3)
    if add_tar_actor == 'on':
        ren.RemoveActor(tar_actor)