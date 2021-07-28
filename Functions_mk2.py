import numpy as np
import vtk
import os
import Functions


# Get point list and size from stl file function
def read_stl(path, name, extention):
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(path+name+extention)
    file_data.Update()
    data = file_data.GetOutput()
    num_points = data.GetNumberOfPoints()
    point_list = []
    for i in range(num_points):
        point_list.append(data.GetPoint(i))
    point_list = np.array(point_list)

    return num_points, point_list, name


# 입력한 array 를 number 의 갯수만큼 x, y, z에 지정한 offset 거리 사이에서 랜덤하게 offset 시키는 함수
# 함수를 한번 불러낼때마다 새로운 offset 이 만들어지므로 하나의 객체에 대한 여러개의 offset 을 만들때 사용
def make_all_offset(number,  x1, x2, y1, y2, z1, z2, array):
    list = []
    temp = []
    x_offset, y_offset, z_offset = Functions.Make_offset_distance(number=number, x1= x1, x2= x2, y1= y1, y2= y2, z1= z1, z2= z2)
    for i in range(number):
        temp.append(x_offset[i])
        temp.append(y_offset[i])
        temp.append(z_offset[i])
        temp = np.array(temp)
        temp2 = np.add(array, temp)
        list.append(temp2)
        temp = []
    return np.array(list)


# Read_stl 의 결과를 dictionary 화 시키는 함수
def property_for_make_training_data(path, name, extension):
    dict = {}
    surface_property = read_stl(path=path, name=name, extention=extension)
    dict['number'] = surface_property[0]
    dict['array'] = surface_property[1]
    dict['name'] = surface_property[2]
    return dict


# Normalize function
def normalize_data(data, number_range, num_end):
    try:
        if data.shape[2] > 0:
            data1 = data[:, 0:number_range, :]
            data2 = data[:, number_range:num_end, :]
    except IndexError as e:
        data1 = data[:, 0:number_range]
        data2 = data[:, number_range:num_end]

    data1_norm = []
    for array in data1:
        element = np.array(array)

        min_val = np.min(array)
        element -= min_val

        max_val = np.max(element)
        element /= max_val
        data1_norm.append(element)
    data1_norm = np.array(data1_norm)

    data2_norm = []
    for array in data2:
        element = np.array(array)

        min_val = np.min(array)
        element -= min_val

        max_val = np.max(element)
        element /= max_val
        data2_norm.append(element)
    data2_norm = np.array(data2_norm)
    result = np.concatenate((data1_norm, data2_norm), axis=1)

    return result


# stl 폴더 내의 set 폴더 내의 stl 파일로 dict 만들기
def make_dict_with_sets(set_number, set_path, call_extension):
    i = 0
    dict = {}
    # path = f'./stl/set{set_number}/'
    path = set_path + f'{set_number}/'
    filelist = os.listdir(path)
    for file in filelist:
        name = file.split('.')[0]
        extension = f'.{file.split(".")[1]}'
        if extension != call_extension:
            continue
        num_points, array, model_name = read_stl(path=path, name=name, extention=extension)
        dict[f'property{i}'] = {}
        dict[f'property{i}']['number of point'] = num_points
        dict[f'property{i}']['array'] = array
        dict[f'property{i}']['name of model'] = model_name
        i += 1
    return dict


# set 리스트를 통해서 dictionary 를 만드는 함수
def make_set_mk2(list_of_set_num, all_property_num, set_path, call_extension):
    set = {}
    for i in range(list_of_set_num.shape[0]):
        center_list = []
        set[list_of_set_num[i]] = make_dict_with_sets(list_of_set_num[i], set_path, call_extension)
        for j in range(all_property_num):
            center_list.append(np.expand_dims(np.average(set[list_of_set_num[i]][f'property{j}']['array'], axis=0), axis=0))
        # center_list.append(np.expand_dims(np.average(set[set_num[i]]['property1']['array'], axis=0), axis=0))
        # center_list.append(np.expand_dims(np.average(set[set_num[i]]['property2']['array'], axis=0), axis=0))
        # center_list.append(np.expand_dims(np.average(set[set_num[i]]['property3']['array'], axis=0), axis=0))
        array_tar = center_list[0]
        flag = 1
        while flag != 0:
            flag = 0
            for j in range(set[list_of_set_num[i]]['property0']['number of point']):
                if array_tar[0][1] > set[list_of_set_num[i]]['property0']['array'][j][1]:
                    array_tar[0][1] -= 1
                    flag += 1
        array_tar[0][1] -= 1
        set[list_of_set_num[i]][f'property{all_property_num}'] = {}
        set[list_of_set_num[i]][f'property{all_property_num}']['number of point'] = array_tar.shape[0]
        set[list_of_set_num[i]][f'property{all_property_num}']['array'] = array_tar
        set[list_of_set_num[i]][f'property{all_property_num}']['name of model'] = 'target'

    return set, center_list


def make_all_rotation_mk2(number, x_angle, y_angle, z_angle, array):
    list = []
    center = -np.average(array, axis=0)

    for i in range(number):
        trans_to_zero = np.add(array, center)

        # x rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select= 'x', angle= x_angle[i])
        rotation1 = np.dot(trans_to_zero, rotation_mat)

        # y rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select='y', angle=y_angle[i])
        rotation2 = np.dot(rotation1, rotation_mat)

        # z rotation
        rotation_mat = Functions.make_rotation_matrix(axis_select='z', angle=z_angle[i])
        rotation3 = np.dot(rotation2, rotation_mat)

        back_to_origin = np.subtract(rotation3, center)
        list.append(back_to_origin)
    return np.array(list)


def make_top_k_average_vertex_mk2(score_data, input_data, k, weight_range1, weight_range2):
    if score_data.shape[0] == 1:
        score_data = np.squeeze(score_data)
    copy_score_data = np.copy(score_data)
    sorted_copy_score_data = np.sort(copy_score_data)[::-1]

    # 가장 점수가 높은 좌표 구하기
    list = []

    for i in range(k):
        index = np.where(sorted_copy_score_data[i] == copy_score_data)[0][0]
        list.append(input_data[index])
    top_k_array = np.array(list)

    # weigth 주기 이건 일단 보류
    # weight = np.linspace(weight_range1, weight_range2, num=k, endpoint=True, retstep=False)

    aver = np.average(top_k_array, axis=0)
    aver = np.expand_dims(aver, axis=0)

    return aver


def make_bot_k_average_vertex_mk2(score_data, input_data, k, weight_range1, weight_range2):
    if score_data.shape[0] == 1:
        score_data = np.squeeze(score_data)
    copy_score_data = np.copy(score_data)
    sorted_copy_score_data = np.sort(copy_score_data)

    # 가장 점수가 낮은 좌표 구하기
    list = []

    for i in range(k):
        index = np.where(sorted_copy_score_data[i] == copy_score_data)[0][0]
        list.append(input_data[index])
    top_k_array = np.array(list)

    # weigth 주기 이건 일단 보류
    # weight = np.linspace(weight_range1, weight_range2, num=k, endpoint=True, retstep=False)

    aver = np.average(top_k_array, axis=0)
    aver = np.expand_dims(aver, axis=0)

    return aver
