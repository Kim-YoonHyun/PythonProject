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
import functions_my_1_2_1 as fmy
import vtk
import functions_vtk as fvtk

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
fmy.data_list_title(f'./data')
data_num = int(input('rendering 할 데이터 번호 선택 > '))
data_path = f'./data/data{data_num}'
with open(f'{data_path}/data_information.pkl', 'rb') as file:
    target_data, bone1_data, bone2_data, skin_data = dill.load(file)

if not os.path.isdir(f'{data_path}/data_image'):
    os.mkdir(f'{data_path}/data_image')

for i in range(target_data.num_of_data):
    # target, bone actor
    target_actor = fvtk.point_actor(target_data.data_vertices_list[i], target_point_size, target_point_color)
    target_actor.SetPosition(0, 0, 0)

    bone1_actor = fvtk.point_actor(bone1_data.data_vertices_list[i], bone1_point_size, bone1_point_color)
    bone1_actor.SetPosition(0, 0, 0)

    bone2_actor = fvtk.point_actor(bone2_data.data_vertices_list[i], bone2_point_size, bone2_point_color)
    bone2_actor.SetPosition(0, 0, 0)

    skin_actor = fvtk.point_actor(skin_data.data_vertices_list[i], skin_point_size, skin_point_color)
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