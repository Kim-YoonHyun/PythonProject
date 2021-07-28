
import vtk
import numpy as np
import os


# set 리스트를 통해서 dictionary 를 만드는 함수
# stl 폴더 내의 set 폴더 내의 stl 파일로 dict 만들기
def read_stl(path, name, extension):
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(path+name+extension)
    file_data.Update()
    data = file_data.GetOutput()
    num_points = data.GetNumberOfPoints()
    point_list = []
    for i in range(num_points):
        point_list.append(data.GetPoint(i))
    point_list = np.array(point_list)

    return num_points, point_list, name


def make_dict_with_sets(set_number, set_path, call_extension):
    i = 0
    dictionary = {}
    path = f'{set_path}{set_number}/'
    file_list = os.listdir(path)
    for file in file_list:
        name = file.split('.')[0]
        extension = f'.{file.split(".")[1]}'
        if extension != call_extension:
            continue
        num_points, array, model_name = read_stl(path=path, name=name, extension=extension)
        dictionary[f'property{i}'] = {}
        dictionary[f'property{i}']['number of point'] = num_points
        dictionary[f'property{i}']['array'] = array
        dictionary[f'property{i}']['name of model'] = model_name
        i += 1
    return dictionary

def make_set(list_of_set_num, all_property_num, set_path, call_extension):
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