import vtk
import functions_vtk

# Rendering camera 관련
position = 20

"""
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
"""

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


# model actor - 0~1 분포도
point_list1 = [
    [1, 2, 3],
    [-2, 3, 5],
    [6, 2, -4]
]
model_actor_distribution = functions_vtk.point_actor(point_list1, point_size1, 'None')
functions_vtk.assign_value_to_polydata(model_actor_distribution.GetMapper().GetInput(), [0, 0.5, 1])
model_actor_distribution.SetPosition(-position, 0, 0)

# model_actor - 특정 색 입히기
point_list2 = [[1, 2, 3]]
model_actor_color = functions_vtk.point_actor(point_list2, point_size2, point_color)
model_actor_color.SetPosition(position, 0, 0)

# add actor
ren.AddActor(model_actor_distribution)
ren.AddActor(model_actor_color)

# renderWindow initialize
renWin.Render()
iren.Initialize()

"""
# 랜더링 이미지 저장
windowToImageFilter = vtk.vtkWindowToImageFilter()
windowToImageFilter.SetInput(renWin)
windowToImageFilter.ReadFrontBufferOff()
windowToImageFilter.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(f"path/filename.png")
writer.SetInputConnection(windowToImageFilter.GetOutputPort())
writer.Write()
"""

"""
# 반복 렌더링이 필요한 경우
# renderWindow.RemoveRenderer(renderer)
ren.RemoveActor(model_actor_distribution)
ren.RemoveActor(model_actor_color)
"""

iren.Start()




