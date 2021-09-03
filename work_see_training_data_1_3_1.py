# sampling 된 data 를 확인하기 위한 랜더링 코드
# score 는 적용되어 있지 않으며 sampling 형상을 확인하기 위함
# (label) 폴더를 구분하는 방식 추가
# data 31번 부터 적용가능

# _0
# work_see_GT_data.py 와 work_see_sampling_data_3 코드를 합치고 선택지를 구성한 코드

# _1
# code renewal

# _1_3_1
# call_data_title 를 사용 20210903
# score map 적용 여부 추가 20210903
# code 구조 재구성 20210903

import numpy as np
import sys
import vtk
import os
import dill
import vtk
import functions_vtk as fvtk
import pandas as pd     # 1.3.1

def call_data_version(path):
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('ver'):
            return file

def call_dataframe(path):
    df = pd.read_csv(f'{path}/data_information.csv')
    df.drop(columns='Unnamed: 0', inplace=True)
    df.fillna('0', inplace=True)
    df.set_index('data_name', inplace=True)
    return df

def call_other_information(path):
    with open(f'{path}/other_information.txt', 'r') as file:
        line = file.readlines()
    other_information = dict(eval('{' + ', '.join(line).replace('\n', '') + '}'))
    return other_information

def call_data_title(path):
    data_list = os.listdir(path)
    for data_name in data_list:
        data_path = f'{path}/{data_name}'
        data_version = call_data_version(data_path)
        input_data = np.load(f'{data_path}/input_data.npy')

        other_information = call_other_information(data_path)
        set_range = other_information['stl set range']

        data_df = call_dataframe(data_path)
        rand_sam = data_df.rand_sam_status[1]
        rot = data_df.rot_status[0]
        trans = data_df.trans_status[0]

        print(f'{data_name:>7s} ({data_version}):{str(input_data.shape):>15s}, {set_range:>8s}, {rand_sam:>8s}, {rot:>7s}, {trans:>18s}')

def call_data_class_information(path):
    with open(f'{path}/data_information.pkl', 'rb') as file:
        return dill.load(file)

def normalize_data(data):
    data_norm = []
    for array in data:
        element = np.array(array)
        min_val = np.min(array)
        element -= min_val

        max_val = np.max(element)
        element = element / max_val
        data_norm.append(element)

    data_norm = np.array(data_norm)

    return data_norm


def rendering():
    global POSITION, SIZE_X, SIZE_Y
    global CLIP_X, CLIP_Y, DIST, FOC_X, FOC_Y, FOC_Z, POS_X, POS_Y, POS_Z, THICK, VU_X, VU_Y, VU_Z
    global TARGET_POINT_SIZE, BONE1_POINT_SIZE, BONE2_POINT_SIZE, SKIN_POINT_SIZE, SCORE
    global TARGET_POINT_COLOR, BONE1_POINT_COLOR, BONE2_POINT_COLOR, SKIN_PIONT_COLOR

    # <vtk 기본 세팅>-----------------------------------------------------------
    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)

    # render window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(int(SIZE_X*3/2), int(SIZE_Y*3/2))

    # Rendering 카메라 위치 설정
    camera = vtk.vtkCamera()
    camera.SetClippingRange(CLIP_X, CLIP_Y)
    camera.SetDistance(DIST)
    camera.SetFocalPoint(FOC_X, FOC_Y, FOC_Z)
    camera.SetPosition(POS_X, POS_Y, POS_Z)
    camera.SetThickness(THICK)
    camera.SetViewUp(VU_X, VU_Y, VU_Z)
    ren.SetActiveCamera(camera)

    # Interactor 설정
    iren = vtk.vtkRenderWindowInteractor()
    interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(interactorStyle)
    iren.SetRenderWindow(renWin)

    # <데이터 불러오기>------------------------------------------
    call_data_title(f'./data')
    data_num = int(input('rendering 할 데이터 번호 선택 > '))
    data_path = f'./data/data{data_num}'
    target_data, bone1_data, bone2_data, skin_data = call_data_class_information(data_path)
    score_distribution = np.load(f'{data_path}/score.npy')

    score_distribution = normalize_data(score_distribution)

    if not os.path.isdir(f'{data_path}/data_image'):
        os.mkdir(f'{data_path}/data_image')

    for i in range(len(target_data.data_vertices_list)):
        # target, bone actor
        target_actor = fvtk.point_actor(target_data.data_vertices_list[i], TARGET_POINT_SIZE, TARGET_POINT_COLOR)
        target_actor.SetPosition(0, 0, 0)

        bone1_actor = fvtk.point_actor(bone1_data.data_vertices_list[i], BONE1_POINT_SIZE, BONE1_POINT_COLOR)
        bone1_actor.SetPosition(0, 0, 0)

        bone2_actor = fvtk.point_actor(bone2_data.data_vertices_list[i], BONE2_POINT_SIZE, BONE2_POINT_COLOR)
        bone2_actor.SetPosition(0, 0, 0)

        if score == 'on':
            skin_actor = fvtk.point_actor(skin_data.data_vertices_list[i], SKIN_POINT_SIZE, 'None')
            fvtk.assign_value_to_polydata(skin_actor.GetMapper().GetInput(), score_distribution[i])
            skin_actor.SetPosition(-POSITION, 0, 0)
        else:
            skin_actor = fvtk.point_actor(skin_data.data_vertices_list[i], SKIN_POINT_SIZE, SKIN_PIONT_COLOR)
            skin_actor.SetPosition(0, 0, 0)




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

# <Rendering camera 관련>-----------------------------------------------
POSITION = 20

# render 카메리 상세 설정시(work_find_rendering_camera 에서 수치 추출 가능)
SIZE_X = 500
SIZE_Y = 500
CLIP_X = 437.48541978551657
CLIP_Y = 832.9353965694179
DIST = 622.2508231247085
FOC_X = -6.5353582725359445
FOC_Y = -34.446804088366704
FOC_Z = 15.888225067695487
POS_X = -28.334831438062928
POS_Y = -653.3622452313335
POS_Z = 76.42372344583254
THICK = 395.4499767839013
VU_X = 0.033045448043591226
VU_Y = -0.09844414645719703
VU_Z = -0.9945937604830992

# point vertex size
TARGET_POINT_SIZE = 5
BONE1_POINT_SIZE = 5
BONE2_POINT_SIZE = 5
SKIN_POINT_SIZE = 5

# point vertex color
# Red, Blue, Green, Yellow, Gray, Dark Gray, Pink, White, Light Red, Black 가능
# 다른 색이 필요하면 Functions.py 내부의 PointActor 함수에 원하는 색깔 추가 가능
TARGET_POINT_COLOR = 'Black'
BONE1_POINT_COLOR = 'Dark Gray'
BONE2_POINT_COLOR = 'Gray'
SKIN_PIONT_COLOR = 'Pink'

SCORE = 'on'

if __name__=='__main__':
    rendering()

exit()