# _2
# 변수를 통해서 rendering 제어

import numpy as np
import vtk
import os
import Functions_tensorflow

#################################################### range setting ####################################################
max_set_num = len(os.walk('./stl/').__next__()[1])


#################################################### parameter #########################################################
# 랜더링 할 set 번호
renderering_set_num = 32

# =======================================================
# Add actor 여부
add_B2_actor = 'onf'
add_B1_actor = 'onf'
add_sur_actor = 'on'
add_D_actor = 'onf'
add_tar_actor = 'ofn'

# =======================================================
# point size & color
B1_size = 5
B2_size = 5
sur_size = 5
D_size = 5
tar_size = 5

# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
B1_color = 'Gray'
B2_color = 'Dark Gray'
sur_color = 'Pink'
D_color = 'Yellow'
tar_color = 'Black'


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

# =========================
# 가지고 있는 모든 set 번호를 list 안에 저장
set_num = []
for i in range(1, max_set_num + 1):
    set_num.append(i)
set_num = np.array(set_num)

# =========================
# property 넘버링
# property0 : Disc
# property1 : Bone1
# property2 : Bone2
# property3 : Skin
target_property_num = 4

#################################################### Make set ##########################################################
set, _ = Functions.make_set(set_num, target_property_num)
print(f"size of {set[renderering_set_num]['property0']['name of model']} : {set[renderering_set_num]['property0']['array'].shape}")
print(f"size of {set[renderering_set_num]['property1']['name of model']} : {set[renderering_set_num]['property1']['array'].shape}")
print(f"size of {set[renderering_set_num]['property2']['name of model']} : {set[renderering_set_num]['property2']['array'].shape}")
print(f"size of {set[renderering_set_num]['property3']['name of model']} : {set[renderering_set_num]['property3']['array'].shape}")
print(f"number of all points = {set[renderering_set_num]['property0']['number of point'] + set[renderering_set_num]['property1']['number of point'] + set[renderering_set_num]['property2']['number of point'] + set[renderering_set_num]['property3']['number of point']}")
print('\n')


################################################### Rendering ##########################################################
# call actor
D_actor = Functions.point_actor(set[renderering_set_num]['property0']['array'], D_size, D_color)
B1_actor = Functions.point_actor(set[renderering_set_num]['property1']['array'], B1_size, B1_color)
B2_actor = Functions.point_actor(set[renderering_set_num]['property2']['array'], B2_size, B2_color)
sur_actor = Functions.point_actor(set[renderering_set_num]['property3']['array'], sur_size, sur_color)
tar_actor = Functions.point_actor(set[renderering_set_num]['property4']['array'], tar_size, tar_color)

# =========================
# Renderer
renderer = vtk.vtkRenderer()

if add_B1_actor == 'on':
    renderer.AddActor(B2_actor)
if add_B2_actor == 'on':
    renderer.AddActor(B1_actor)
if add_sur_actor == 'on':
    renderer.AddActor(sur_actor)
if add_D_actor == 'on':
    renderer.AddActor(D_actor)
if add_tar_actor == 'on':
    renderer.AddActor(tar_actor)

renderer.SetBackground(1,1,1)

# =========================
# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(size_x, size_y)

# =========================
# Render window interactor
iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)

renWin.Render()

# =========================
# Rendering 카메라 위치 설정
camera = vtk.vtkCamera()
camera.SetPosition(pos_x, pos_y, pos_z)
camera.SetFocalPoint(foc_x, foc_y, foc_z)
camera.SetViewUp(view_x, view_y, view_z)
camera.SetClippingRange(clip_x, clip_y)

renderer.SetActiveCamera(camera)

windowToImageFilter = vtk.vtkWindowToImageFilter()
windowToImageFilter.SetInput(renWin)
windowToImageFilter.ReadFrontBufferOff()
windowToImageFilter.Update()

renWin.Render()
iren.Start()


################################################### make flag folder ###################################################
flag = 'off'
try:
    if not (os.path.isfile(f'./stl/set{renderering_set_num}/1; information.txt')):
        flag = 'on'
except OSError as e:
    if e.errno != e.errno.EEXIST:
        raise

if flag == 'on':
    file = open(f'./stl/set{renderering_set_num}/1; information.txt', 'w', encoding= 'utf8')
    line = f"<set{renderering_set_num}>\n" \
        f"size of {set[renderering_set_num]['property0']['name of model']} : {set[renderering_set_num]['property0']['array'].shape}\n" \
        f"size of {set[renderering_set_num]['property1']['name of model']} : {set[renderering_set_num]['property1']['array'].shape}\n" \
        f"size of {set[renderering_set_num]['property2']['name of model']} : {set[renderering_set_num]['property2']['array'].shape}\n" \
        f"size of {set[renderering_set_num]['property3']['name of model']} : {set[renderering_set_num]['property3']['array'].shape}\n" \
        f"number of all points = {set[renderering_set_num]['property0']['number of point'] + set[renderering_set_num]['property1']['number of point'] + set[renderering_set_num]['property2']['number of point'] + set[renderering_set_num]['property3']['number of point']}\n\n" \

    file.write(line)
    print('\n\n추가 정보 기입')
    string = str(input())
    file.write(f'추가 정보: {string}')

    file.close()






