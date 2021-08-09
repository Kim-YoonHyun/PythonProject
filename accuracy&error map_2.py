import vtk
import numpy as np
import Functions
import Functions_mk2
import Functions_mk3


model_point_size = 10
gt_point_size = 10
position = 130

data_number = 33
Multi_Targeting = 'asub'

# train, val, val
val_percentage = 20
test_percentage = 20

size_x = 1134
size_y = 677
clip_x = 665.1710748395149
clip_y = 1172.2122690913402
Dist = 911.0374301368666
foc_x = -3.069484594097492
foc_y = -23.798211327444896
foc_z = 75.8143013193692
pos_x = 27.93590148221489
pos_y = -911.0651760143764
pos_z = 280.22858291408596
Thick = 507.04119425182535
VU_x = -0.13008260787722775
VU_y = -0.22691212760424814
VU_z = -0.9651887905865737


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


optimal_score = 0.0

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
print(test_GT_data.shape)
norm_test_GT_data = Functions_mk3.normalize_data(test_GT_data)
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
    reverse_norm_error_map = np.abs(np.subtract(norm_error_map, 1))
    error_actor = Functions.point_actor(skin_surface, 7, 'None')
    Functions.assign_value_to_polydata(error_actor.GetMapper().GetInput(), reverse_norm_error_map[0])
    error_actor.SetPosition(0, 0, 180)

    predicted_min_vertex = Functions_mk2.make_bot_k_average_vertex_mk2(test_output,
                                                                       test_data[:, 0:num_skin_points, 0:3][
                                                                           i], 1,
                                                                       1, 1)
    norm_test_GT_data_skin = norm_test_GT_data[i][0:num_skin_points]
    optimal_point_index = np.array(np.where(norm_test_GT_data_skin == optimal_score))[0][0]
    gt_score_min_vertex = skin_surface[optimal_point_index]
    pmv_D = Functions.euclidean_distance(predicted_min_vertex, gt_score_min_vertex)

    model_pmv_actor = Functions.point_actor(predicted_min_vertex, 10, "Gray")
    model_pmv_actor.SetPosition(position, 0, 0)

    # gt_actor
    reverse_norm_test_GT_data_skin = np.abs(np.subtract(norm_test_GT_data_skin, 1))
    gt_actor = Functions.point_actor(skin_surface, 7, 'None')
    Functions.assign_value_to_polydata(gt_actor.GetMapper().GetInput(), reverse_norm_test_GT_data_skin)
    gt_actor.SetPosition(-position, 0, 0)

    # gt entry actor
    gt_entry_actor1 = Functions.point_actor([gt_score_min_vertex], 10, "Black")
    gt_entry_actor1.SetPosition(position, 0, 0)
    gt_entry_actor2 = Functions.point_actor([gt_score_min_vertex], 10, "Black")
    gt_entry_actor2.SetPosition(-position, 0, 0)

    # model actor
    reverse_test_output = np.abs(np.subtract(test_output, 1))
    model_actor = Functions.point_actor(skin_surface, 7, 'None')
    Functions.assign_value_to_polydata(model_actor.GetMapper().GetInput(), reverse_test_output)
    model_actor.SetPosition(position, 0, 0)

    # add actor
    ren.AddActor(error_actor)
    ren.AddActor(gt_actor)
    ren.AddActor(model_actor)
    ren.AddActor(gt_entry_actor1)
    ren.AddActor(gt_entry_actor2)
    ren.AddActor(model_pmv_actor)



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
    ren.RemoveActor(gt_actor)
    ren.RemoveActor(gt_entry_actor1)
    ren.RemoveActor(gt_entry_actor2)
    ren.RemoveActor(model_actor)
    ren.RemoveActor(model_pmv_actor)
