
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd
import functions_my_1_2_1 as fmy
import vtk
import functions_vtk as fvtk
import Augmentation_1_2_1 as Aug


with open(f'./data/data1/data_information.pkl', 'rb') as file:
    target_data, bone1_data, bone2_data, skin_data = dill.load(file)
print(target_data)

offset = fmy.make_trans_offset(54, 20, [-5., 5.], [-5., 5.], [-5., 5.])
print(np.array(offset).shape)

target_data.translate(offset, 20)
print(target_data)

temp_list = []
all_b1b2s_data = [bone1_data, bone2_data, skin_data]
for data in all_b1b2s_data:
    temp_list.append(data.data_vertices_list)
temp_array = np.concatenate(temp_list, axis=1)
print(temp_array.shape)
print(temp_array[0].shape)

aaa = []
for i in range(54):
    temp_array2 = np.tile(temp_array[i], [20, 1, 1])
    multi_tar = np.array(target_data.data_vertices_list)[0+20*i: 20+20*i, :, :]
    multi_data = np.concatenate((temp_array2, multi_tar), axis=1)
    aaa.append(multi_data)
result = np.concatenate(aaa, axis=0)
print(result.shape)
print(result[0][0], result[0][1900])
print(result[19][0], result[19][1900])
print(result[20][0], result[20][1900])
print(result[39][0], result[39][1900])

exit()




# point vertex size
point_size1 = 10
point_size2 = 3

# point vertex color
# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
point_color = 'Red'


##### vtk 기본 세팅
# renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(500, 500)

"""
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
"""

# Interactor 설정
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)
point_lists = np.load('./skin_array.npy')
i = 0

# 반복 렌더링 시
while True:

    if i == point_lists.shape[0]:
        i = 0
    # model_actor - 특정 색 입히기
    model_actor_color = fvtk.point_actor(point_lists[i], point_size2, point_color)
    model_actor_color.SetPosition(position, 0, 0)

    # add actor
    ren.AddActor(model_actor_color)

    # renderWindow initialize
    renWin.Render()
    iren.Initialize()


    # 랜더링 이미지 저장 (이 위치에서만 가능)
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f"./skin_array/{i}.png")
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    # 반복 렌더링이 필요한 경우
    iren.Start()  # 반복 중간중간 제어가 필요할 때 사용
    ren.RemoveActor(model_actor_color)

    i += 1






