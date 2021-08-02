
import vtk
import numpy as np
import os

def read_stl(path, stl_name):
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(f'{path}/{stl_name}.stl')
    file_data.Update()
    num_of_stl_points = file_data.GetOutput().GetNumberOfPoints()
    stl_vertices_list = list(
        list(file_data.GetOutput().GetPoint(i)) for i in range(num_of_stl_points)
    )
    return stl_vertices_list, num_of_stl_points


def up_sampling(array, unity_number, max_length):
    up_sampled_lists = []

    # point 생성
    # array 순서대로 진행
    for idx in range(array.shape[0]):

        start = time.time()
        num_matchs = np.array([[0, 0]])
        count = 0

        # 중복되지 않는 숫자 짝 만들기.
        while unity_number - array[idx].shape[0] > num_matchs.shape[0]-1:

            # 랜덤 point 선택, 나머지 모든 point 와 거리계산
            place = []
            numrange = array[idx].shape[0]
            random_nums = np.array([random.randrange(0, numrange)])
            length = euclidean_distance(array[idx][random_nums], array[idx])
            sorted_length = np.sort(length)

            # max_length 이상 떨어져 있는 point들 중에서 가장 가까운 점 선택.
            flag = 0
            order = 0
            while flag < 1:
                if sorted_length[order] > max_length:
                    place.append(np.where(sorted_length[order] == length)[0][0])
                    flag += 1
                    count += 1
                order += 1
            place = np.expand_dims(np.array(place), axis=1)

            # 랜덤 선택한 점과 일정거리 이상 떨어져 있는 점의 짝이 중복되지 않도록 조정
            num_match = np.concatenate((np.array([random_nums]), place), axis=1)
            sorted_num_match = np.sort(num_match)
            try:
                np.where(sorted_num_match == num_matchs)[0][0]
                num_matchs = num_matchs
            except IndexError as e:
                num_matchs = np.concatenate((num_matchs, sorted_num_match), axis=0)

            # 무한 루프 해결중
            if count - (num_matchs.shape[0] - 1) == 5000:
                print('real:', num_matchs.shape[0] - 1)
                training_time, hour, minute, sec = record_time(start)
                print('run time :', training_time, 'sec')
                count = num_matchs.shape[0] - 1
                max_length += 0.2

        # 만들어진 숫자 짝 리스트를 통해서 새로운 point 생성, 기존의 array에 추가
        num_matchs = num_matchs[1:, 0:] # 첫 번째 차원에선 1부터, 두 번째 차원에선 0부터 남김
        num_matchs = np.transpose(num_matchs)
        center_points = np.take(array[idx], num_matchs[0], axis=0)
        away_points = np.take(array[idx], num_matchs[1], axis=0)
        # 생성
        new_points = (center_points + away_points) / 2
        # 추가
        up_sampled_array = np.concatenate((array[idx], new_points), axis=0)
        up_sampled_lists.append(up_sampled_array)

    up_sampled_arrays = np.array(up_sampled_lists)

    return up_sampled_arrays


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