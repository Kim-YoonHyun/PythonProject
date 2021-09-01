
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd
import functions_my_1_3 as fmy
import vtk
import functions_vtk as fvtk
import Augmentation_1_3 as Aug


a = [3, 4]
b = [5, 6]
c = [a, b]
print(c)
a.append(7)
print(c)
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






