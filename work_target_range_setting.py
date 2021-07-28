# see_case_vtk_2 와 완전히 똑같은 code
# score가 적용된 학습용 data set의 랜더링을 볼 수 있는 code
# data 29번 부터 적용가능
# train 과 test data 가 따로 나뉘어져 있지 않은 data를 보는 code
# _3
# 불러오는 데이터 변경

import numpy as np
import Functions_tensorflow
import vtk
import os
import Functions_mk2_tensorflow



#################################################### parameter #########################################################
tar_x = [-18.0, 18.0]
tar_y = [-3, 7.0]
tar_z = [-2.0, 2.0]

tar_increasing_number = 4000

# =======================================================
# set 번호 범위 지정
set_num = []
for i in range(32, 33):
    set_num.append(i)
set_num = np.array(set_num)

# 지정한 범위의 stl 전부 불러오기
set, _ = Functions_mk2_tensorflow.make_set_mk2(set_num, 4, './stl/set', '.stl')

# 모든 set 의 model 정보 print
for i in range(set_num.shape[0]):
    print(f"size of {set[set_num[i]]['property0']['name of model']} : {set[set_num[i]]['property0']['array'].shape}")
    print(f"size of {set[set_num[i]]['property1']['name of model']} : {set[set_num[i]]['property1']['array'].shape}")
    print(f"size of {set[set_num[i]]['property2']['name of model']} : {set[set_num[i]]['property2']['array'].shape}")
    print(f"size of {set[set_num[i]]['property3']['name of model']} : {set[set_num[i]]['property3']['array'].shape}")
    print(f"number of all points = {set[set_num[i]]['property0']['number of point'] + set[set_num[i]]['property1']['number of point'] + set[set_num[i]]['property2']['number of point'] + set[set_num[i]]['property3']['number of point']}")
    print('\n')

disc = set[32]['property0']['array']
center = np.expand_dims(np.average(disc, axis=0), axis=0)
center_y = center[0][1]

min_disc = np.min(disc, axis=0)[1]

while(center_y > min_disc):
    center_y -= 1
new_center = np.array([center[0][0], center_y, center[0][2]])
center = np.expand_dims(new_center, axis=0)

x_offset, y_offset, z_offset = Functions.make_offset_distance(tar_increasing_number, tar_x[0], tar_x[1], tar_y[0], tar_y[1], tar_z[0], tar_z[1])
all_Target = Functions.make_all_offset(center, x_offset, y_offset, z_offset, tar_increasing_number - 1)
all_Target = np.squeeze(all_Target)
print(all_Target.shape)

disc_point_size = 6
center_point_size = 4
all_target_point_size = 4

# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가\능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
disc_point_color = 'Yellow'
center_point_color = 'Red'
all_target_point_color = 'Red'

# =======================================================
# Renderer
renderer = vtk.vtkRenderer()

# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(1000, 1000)

# Render window interactor
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)


# call actor
disc_actor = Functions.point_actor(disc, disc_point_size, disc_point_color)
center_actor = Functions.point_actor(center, center_point_size, center_point_color)
all_target_actor = Functions.point_actor(all_Target, all_target_point_size, all_target_point_color)

# add actor
renderer.AddActor(disc_actor)
renderer.AddActor(center_actor)
renderer.AddActor(all_target_actor)

renderer.SetBackground(1,1,1)

renWin.Render()
iren.Start()








