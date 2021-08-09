import vtk
import numpy as np
import Functions
import Functions_mk3


model_point_size = 10
gt_point_size = 10
position = 130

data_number = 33
Multi_Targeting = 'asub'

# train, val, val
val_percentage = 20
test_percentage = 20

size_x = 1050
size_y = 535
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


# ############################## train/val set ########################################################################
# path 설정
data_path = './data/data' + str(data_number)
test_model_name = 'test_37_model_Net_23_Seq_3_Ch_256_Ker_1'

# load array
if Multi_Targeting == 'sub':
    data = np.load(data_path + f'/MT_data_sub.npy')
elif Multi_Targeting == 'concat':
    data = np.load(data_path + f'/MT_data_concat.npy')
elif Multi_Targeting == 'sub_concat':
    data = np.load(data_path + f'/MT_data_sub_concat.npy')
else:
    data = np.load(data_path + f'/data_labeled.npy')
GT_data = np.load(data_path + f'/score.npy')
num_skin_points = np.load(data_path + f'/num_skin_points.npy')

# variables
num_all_points = GT_data.shape[1]
num_all_data = GT_data.shape[0]
num_test_data = int(num_all_data * test_percentage/100)
num_val_data = int((num_all_data - num_test_data) * val_percentage/100)
num_train_data = num_all_data - num_val_data - num_test_data

# data 분류
train_data = []
train_GT_data = []
val_data = []
val_GT_data = []
test_data = []
test_GT_data = []

for i in range(num_all_data):
    if i < num_train_data:
        train_data.append(data[i])
        train_GT_data.append(GT_data[i])
    elif i >= num_train_data and i < num_train_data + num_val_data:
        val_data.append(data[i])
        val_GT_data.append(GT_data[i])
    else:
        test_data.append(data[i])
        test_GT_data.append(GT_data[i])

test_data = np.array(test_data)
test_GT_data = np.array(test_GT_data)

calcul_pass = './test_result/test_37_model_Net_23_Seq_3_Ch_256_Ker_1/D & loss/'


# ############################## Rendering #############################################################################
# renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)

# render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(int(size_x*3/2), int(size_y*3/2))

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

iren = vtk.vtkRenderWindowInteractor()
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)

test_GT_data_norm = Functions_mk3.normalize_data(test_GT_data)
for i in range(len(test_GT_data)):
    test_output = np.load(calcul_pass + str(i+1) + '_test_output.npy')
    test_output_aver = np.average(test_output)
    test_GT_data_norm_aver = np.average(test_GT_data_norm[i])
    error_map = np.abs(np.subtract(test_GT_data_norm[i], test_output))
    accuracy = 100*(1-abs(test_GT_data_norm_aver - test_output_aver)/test_GT_data_norm_aver)
    print(accuracy)

    skin_surface = test_data[i][:1000, :3]

    norm_error_map = Functions_mk3.normalize_data([error_map])
    error_actor = Functions.point_actor(skin_surface, gt_point_size, 'None')
    Functions.assign_value_to_polydata(error_actor.GetMapper().GetInput(), norm_error_map[0])
    error_actor.SetPosition(0, 0, 0)

    # add actor
    ren.AddActor(error_actor)

    # renderWindow initialize
    renWin.Render()
    iren.Initialize()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    # 랜더링 이미지 저장
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(f'./test_result/{test_model_name}/norm error map/index{i+1}_acc{round(accuracy, 2)}.png')
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    ren.RemoveActor(error_actor)

