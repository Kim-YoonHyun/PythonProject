import numpy as np
import vtk
import random
import math
import time
import Functions_mk2
import os
import torch


cuda = torch.device('cuda')


# ############################## Functions #############################################################################
# 시간기록용 함수
def record_time(start):
    all_time = round(time.time() - start, 2)
    hour = all_time // 3600
    minute = (all_time % 3600) // 60
    sec = all_time % 60

    return all_time, int(hour), int(minute), round(sec, 2)


# Get point list and size from stl file function
def read_stl(name):
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(name)
    file_data.Update()
    data = file_data.GetOutput()
    num_points = data.GetNumberOfPoints()
    point_list = []
    for i in range(num_points):
        point_list.append(data.GetPoint(i))

    return num_points, np.array(point_list), name


# x, y, z 방향으로 offset 거리를 생성하는 function
def make_offset_distance(number, x1, x2, y1, y2, z1, z2):
    x = []
    y = []
    z = []
    for i in range(number):
        x_offset = random.uniform(x1, x2)
        y_offset = random.uniform(y1, y2)
        z_offset = random.uniform(z1, z2)
        x.append(round(x_offset, 5))
        y.append(round(y_offset, 5))
        z.append(round(z_offset, 5))
    return np.array(x), np.array(y), np.array(z)


# x, y, z 방향으로 회전시킬 각도를 생성하는 함수
def make_rotate_angle(number, x1, x2, y1, y2, z1, z2):
    x_degree = []
    y_degree = []
    z_degree = []
    for i in range(number):
        x_degree.append(random.uniform(x1, x2) * (math.pi / 180))
        y_degree.append(random.uniform(y1, y2) * (math.pi / 180))
        z_degree.append(random.uniform(z1, z2) * (math.pi / 180))

    return np.array(x_degree), np.array(y_degree), np.array(z_degree)


# x, y, z 중 하나의 축을 선택하여 해당 축을 기준으로 회전을 시킬 수 있는 matrix 를 만들어 주는 함수
def make_rotation_matrix(axis_select, angle):
    if axis_select == 'x':
        matrix = np.array([[1.,               0.,               0.],
                           [0.,  math.cos(angle), -math.sin(angle)],
                           [0.,  math.sin(angle),  math.cos(angle)]])
    elif axis_select == 'y':
        matrix = np.array([[math.cos(angle),  0.,  math.sin(angle)],
                           [0.,               1.,               0.],
                           [-math.sin(angle), 0.,  math.cos(angle)]])
    elif axis_select == 'z':
        matrix = np.array([[math.cos(angle), -math.sin(angle),  0.],
                           [math.sin(angle),  math.cos(angle),  0.],
                           [0.,               0.,               1.]])
    else:
        matrix = None
    return matrix


# 선택한 축을 기준으로 입력한 array를 입력한 angle 만큼 회전 시킨 array 를 number 만큼 얻어내는 함수
def make_all_rotation(number, x1, x2, y1, y2, z1, z2, array, center):
    answer_list = []
    x_angle, y_angle, z_angle = make_rotate_angle(number=number, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)
    for i in range(number):
        trans_to_zero = np.add(array, center)

        # x rotation
        rotation_mat = make_rotation_matrix(axis_select='x', angle=x_angle[i])
        rotation1 = np.dot(trans_to_zero, rotation_mat)

        # y rotation
        rotation_mat = make_rotation_matrix(axis_select='y', angle=y_angle[i])
        rotation2 = np.dot(rotation1, rotation_mat)

        # z rotation
        rotation_mat = make_rotation_matrix(axis_select='z', angle=z_angle[i])
        rotation3 = np.dot(rotation2, rotation_mat)

        back_to_origin = np.subtract(rotation3, center)

        answer_list.append(back_to_origin)
    return np.array(answer_list)


# 입력한 데이터를 입력한 offset 의 갯수만큼 만드는 function
# 이미 정해져있는 offset 거리를 적용하여 객체를 offset 시키므로 여러 객체를 동시에 같은 거리로 offset 시킬 때 사용
def make_all_offset(array, x_offset, y_offset, z_offset, number):
    answer_ist = []
    temp = []
    for i in range(number):
        temp.append(x_offset[i])
        temp.append(y_offset[i])
        temp.append(z_offset[i])
        temp = np.array(temp)
        temp2 = np.add(array, temp)
        answer_ist.append(temp2)
        temp = []
    return np.array(answer_ist)


# vector 의 크기를 구하는 function
def abs_vector(vertex):
    value = np.sqrt(np.sum(np.square(vertex), axis=-1))
    return value


# 두 점 사이의 거리를 구하는 function
def euclidean_distance(start, end):
    value = abs_vector(np.subtract(start, end))

    return value


# 전체 입력값에서 원하는 범위만큼 뽑아내는 함수
def pick_up_array(case, case_number,  range1, range2):
    answer_list = []
    for i in range(range1, range2):
        answer_list.append(case[case_number][i])

    return np.array(answer_list)


#  두 점과 하나의 중심점으로 만들어진 두 개의 직선 사이의 각도값을 계산하는 함수
def calculate_theta(vertex1, vertex2, center):
    v1 = np.subtract(vertex1, center)
    v2 = np.subtract(vertex2, center)
    internal = np.sum(np.multiply(v1, v2), axis= -1)
    abs_v1 = np.sqrt(np.sum(np.square(v1), axis= -1))
    abs_v2 = np.sqrt(np.sum(np.square(v2), axis= -1))
    mul_v1v2 = np.multiply(abs_v1, abs_v2)
    theta = np.arccos(np.divide(internal, mul_v1v2))

    return np.array(theta)


# axis 1 에 존재하는 array 를 전부 합친 다음 원하는 수치만큼 array 만들어 그 평균치를 재분배 하는 함수
def make_average_array(array, batch, block_number):
    array_sum = np.sum(array, axis=1, keepdims=True)
    output = []
    for i in range(batch):
        temp = []
        average_num = np.divide(array_sum[i][0], 2)
        for j in range(block_number):
            temp.append(average_num)
        output.append(temp)

    return np.array(output)


# 입력한 point cloud 로 Actor 를 만드는 함수
def point_actor(point_cloud, point_size, color):
    color_preset = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [160, 160, 160],
        [255, 51, 204],
        [255, 255, 255],
        [90, 90, 90],
        [255, 100, 100],
        [0, 0, 0]
    ]
    output_points = vtk.vtkPoints()
    output_vertices = vtk.vtkCellArray()
    output_scalars = vtk.vtkFloatArray()
    output_scalars.SetNumberOfComponents(1)
    output_scalars.SetName("Sequence")

    output_colors = vtk.vtkUnsignedCharArray()
    output_colors.SetNumberOfComponents(3)
    output_colors.SetName("Colors")

    for point in point_cloud:
        id = output_points.InsertNextPoint(point[:3])
        output_vertices.InsertNextCell(1)
        output_vertices.InsertCellPoint(id)
        output_scalars.InsertNextTuple([1.0])
        if color == 'Red':
            output_colors.InsertNextTuple(color_preset[0])
        if color == 'Blue':
            output_colors.InsertNextTuple(color_preset[1])
        if color == 'Green':
            output_colors.InsertNextTuple(color_preset[2])
        if color == 'Yellow':
            output_colors.InsertNextTuple(color_preset[3])
        if color == 'Double':
            output_colors.InsertNextTuple(color_preset[int(point[3])])
        if color == 'Gray':
            output_colors.InsertNextTuple(color_preset[4])
        if color == 'Pink':
            output_colors.InsertNextTuple(color_preset[5])
        if color == 'White':
            output_colors.InsertNextTuple(color_preset[6])
        if color == 'Dark Gray':
            output_colors.InsertNextTuple(color_preset[7])
        if color == 'Light Red':
            output_colors.InsertNextTuple(color_preset[8])
        if color == 'Black':
            output_colors.InsertNextTuple(color_preset[9])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(output_points)  # Point data 를 입력
    polydata.SetVerts(output_vertices)  # vertex 정보를 입력
    if color == 'None':
        polydata.GetPointData().SetScalars(output_scalars)  # 위에서 지정한 색 입력
    else:
        polydata.GetPointData().SetScalars(output_colors)  # 위에서 지정한 색 입력
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)

    return actor


# Normalize function
def normalize_data(data):
    result = []
    for array in data:
        element = np.array(array)

        min_val = np.min(array)
        element -= min_val

        max_val = np.max(element)
        element /= max_val
        result.append(element)
    result = np.array(result)

    return result


# value assign function
def assign_value_to_polydata(polydata, scalar_data):
    for idx, scalar in enumerate(scalar_data):
        polydata.GetPointData().GetScalars().SetTuple(idx, [scalar])

    polydata.GetPointData().Modified()


# 퍼센티지 바 출력하는 함수
def make_percentage_bar(percentage, bar_length):
    one_block = 100/bar_length
    datum_point = int(percentage/one_block)
    bar1 = '|'
    bar2 = ''
    for i in range(1, bar_length + 1):
        if i <= datum_point:
            bar1 += '='
            if i % int(bar_length/10) == 0:
                bar1 += '|'
        else:
            bar2 += '-'
            if i % int(bar_length/10) == 0:
                bar2 += '|'
    return bar1 + bar2


# 폴더 내부의 STL 파일을 불러와서 dictionary 를 만드는 함수
def read_stl_in_folder(number):
    i = 0
    dictionary = {}
    path = f'./stl/set{number}/'
    file_list = os.listdir(path)
    for file in file_list:
        name = file.split('.')[0]
        extension = f'.{file.split(".")[1]}'
        if extension != '.stl':
            continue
        num_points, array, model_name = Functions_mk2.ReadSTL(path=path, name=name, extention=extension)
        dictionary[f'property{i}'] = {}
        dictionary[f'property{i}']['number of point'] = num_points
        dictionary[f'property{i}']['array'] = array
        dictionary[f'property{i}']['name of model'] = model_name
        i += 1
    return dictionary


# set 리스트를 통해서 dictionary 를 만드는 함수
def make_set(set_num, target_property_num):
    set = {}
    for i in range(set_num.shape[0]):
        center_list = []
        set[set_num[i]] = read_stl_in_folder(set_num[i])
        center_list.append(np.expand_dims(np.average(set[set_num[i]]['property0']['array'], axis=0), axis=0))
        center_list.append(np.expand_dims(np.average(set[set_num[i]]['property1']['array'], axis=0), axis=0))
        center_list.append(np.expand_dims(np.average(set[set_num[i]]['property2']['array'], axis=0), axis=0))
        center_list.append(np.expand_dims(np.average(set[set_num[i]]['property3']['array'], axis=0), axis=0))
        array_tar = center_list[0]
        flag = 1
        while (flag != 0):
            flag = 0
            for j in range(set[set_num[i]]['property0']['number of point']):
                if array_tar[0][1] > set[set_num[i]]['property0']['array'][j][1]:
                    array_tar[0][1] -= 1
                    flag += 1
        array_tar[0][1] -= 1
        set[set_num[i]][f'property{target_property_num}'] = {}
        set[set_num[i]][f'property{target_property_num}']['number of point'] = array_tar.shape[0]
        set[set_num[i]][f'property{target_property_num}']['array'] = array_tar
        set[set_num[i]][f'property{target_property_num}']['name of model'] = 'target'

    return set, center_list


# target 에서 각 파트간의 모든 point 에 대한 길이의 평균값을 구하고 해당 값보다 멀리 있는 point 를 제거
def calculate_aver_array(set_num, all_tar_array, all_array):
    all_aver_array = []
    for i in range(set_num.shape[0]):
        line = euclidean_distance(all_tar_array[i], all_array[i])
        aver = np.average(line)
        temp = []
        for j in range(all_array[i].shape[0]):
            if line[j] < aver:
                temp.append(all_array[i][j])
        temp = np.array(temp)
        all_aver_array.append(temp)
    all_aver_array = np.array(all_aver_array)

    return all_aver_array


# calculate_aver_array 를 통해 만들어진 array 를 기반으로 up sampling 하는 함수
def aver_sampling(array, unity_number, k_number, max_length):
    old_points = []

    # point 생성
    for idx in range(array.shape[0]):
        num_range = array[idx].shape[0]
        random_nums = np.array([], dtype=np.int32)
        while random_nums.shape[0] < unity_number - array[idx].shape[0]:
            random_nums = np.concatenate((random_nums, np.array([random.randrange(0, num_range)])), axis=0)
            random_nums = np.unique(random_nums)

        for z in range(random_nums.shape[0]):
            copy_array = array[idx][random_nums[z]]
            length = euclidean_distance(copy_array, array[idx])
            sorted_length = np.sort(length)

            place = []
            count = 0
            order = 0
            while count < k_number:
                if sorted_length[order] > max_length:
                    place.append(np.where(sorted_length[order] == length)[0][0])
                    count += 1
                order += 1

            # 추가할 point 갯수가 최종 목표치를 넘어설 경우
            temp = array[idx].shape[0] + count
            if temp > unity_number:
                while temp > unity_number:
                    count -= 1
                    temp = array[idx].shape[0] + count

            new_points = []
            for i in range(count):
                new_point = np.add(array[idx][random_nums[z]], array[idx][place[i]]) / 2
                # for j in range(len(old_points)):
                #     temp = new_point - old_points[j]
                # print('\n')
                new_points.append(new_point)
            new_points = np.array(new_points)

            for i in range(len(new_points)):
                array[idx] = np.concatenate((array[idx], np.expand_dims(new_points[i], axis=0)), axis=0)

            for i in range(new_points.shape[0]):
                old_points.append(np.squeeze(new_points[i]))

    flag = 1
    temp = np.expand_dims(array[0],axis=0)
    while flag < array.shape[0]:
        temp = np.concatenate((temp, np.expand_dims(array[flag],axis=0)))
        flag += 1
    final_array = temp

    return final_array


# random sampling 함수
def random_sampling(array, dummy_array, rand_sam_num):
    target_num = []
    all_rand_arrays = dummy_array
    for idx in range(array.shape[0]):
        # rand sam num 갯수만큼 뽑아내기
        extracted_lists = []

        # 기존의 array를 보존하기 위해 복사
        copy_array = array[idx].copy()
        while(copy_array.shape[0] >= rand_sam_num):
            rand_indices = np.array([], dtype=np.int32)

            # 중복되지 않게 랜덤 point 번호 선택
            while(rand_indices.shape[0] < rand_sam_num):
                rand_index = random.randrange(0, copy_array.shape[0])
                rand_indices = np.concatenate((rand_indices, np.array([rand_index])), axis=0)
                rand_indices = np.unique(rand_indices)

            # rand sam num 만큼 point를 뽑아낸 array
            extracted_array = np.take(copy_array, rand_indices, axis=0)
            extracted_lists.append(extracted_array)

            # 추출하고 남은 array
            copy_array = np.delete(copy_array, rand_indices, axis=0)

        # 추출한 모든 random sampling array를 모아둔 arrays
        extracted_arrays = np.array(extracted_lists)
        target_num.append(extracted_arrays.shape[0] + 1)

        # 정확히 나뉘어 떨어지지 않고 남아있는 point 가 있는지 확인
        lack_num = rand_sam_num - copy_array.shape[0]
        # 만약 정확히 나뉘어 떨어질 경우
        if lack_num == rand_sam_num:
            target_num[idx] -= 1
            all_rand_arrays = np.concatenate((all_rand_arrays, extracted_arrays), axis=0)
            print('done')
            continue

        # 남은 point 가 부족해서, 부족한 만큼 이미 추출된 point 에서 충당하여 random sampling
        rand_indices = []
        total = 0
        for i in range(extracted_arrays.shape[0]):
            rand_indices.append([])

        # 이미 추출된 point 들 중 부족한 갯수 만큼 pick up
        while(total != lack_num):
            total = 0
            rand_pick1 = random.randrange(0, extracted_arrays.shape[0])
            rand_pick2 = random.randrange(0, extracted_arrays.shape[1])
            # rand_indices[rand_pick1].append(extracted_arrays[rand_pick1][rand_pick2])

            rand_indices[rand_pick1].append(rand_pick2)
            for i in range(extracted_arrays.shape[0]):
                rand_indices[i] = list(np.unique(rand_indices[i]))
                total += len(rand_indices[i])
        rand_indices = np.array(rand_indices)

        # pick up 한 point 들을 부족했던 array에 충당
        for i in range(rand_indices.shape[0]):
            temp = np.take(extracted_arrays[i], rand_indices[i], axis=0)
            copy_array = np.concatenate((copy_array, temp), axis=0)

        # 부족한 문제를 없앤 마지막 array를 최종적으로 통합
        extracted_arrays = np.concatenate((extracted_arrays, np.expand_dims(copy_array, axis=0)), axis= 0)
        all_rand_arrays = np.concatenate((all_rand_arrays, extracted_arrays), axis=0)

    # concatenate를 위해 만들어둔 더미 array 제거.
    all_rand_arrays = np.delete(all_rand_arrays, 0, axis=0)

    return all_rand_arrays, target_num


# 3차원 공간 상에서 center에서 target array 로 이어지는 line과 start array 간의 거리를 구하는 함수
def calculate_point_to_line_length(center_array, start_array, target_array):
    tar_to_start_array_vector = np.subtract(start_array, center_array)
    tar_to_target_array_vector = np.subtract(target_array, center_array)

    L = []
    for i in range(target_array.shape[0]):
        dot_product = np.dot(tar_to_start_array_vector, tar_to_target_array_vector[i])
        abs_dot_product = np.absolute(dot_product)

        cross_temp1 = abs_vector(tar_to_start_array_vector)
        cross_temp2 = abs_vector(tar_to_target_array_vector[i])
        abs_cross_product = cross_temp1 * cross_temp2
        vec_between_vec_radian = np.arccos(abs_dot_product / abs_cross_product)
        vec_between_vec_degree = vec_between_vec_radian * (180 / math.pi)

        len_of_start_array_and_center = euclidean_distance(center_array, start_array)
        L_set = np.multiply(len_of_start_array_and_center, np.sin(vec_between_vec_radian))
        L.append(min(L_set))
    L = np.array(L)
    return L


# Up sampling 함수
# array 는 (10, 1000, 3) 의 형태로 넣으면 결과값이 (10, unity_number, 3) 으로 나온다.
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


# model list title making function
def model_list_title(file_path, row_num, string_length, blank_num):
    file_list = os.listdir(file_path)

    row_num = row_num
    column_num = len(file_list) // row_num
    extra = len(file_list) % row_num

    string_length = string_length
    blank_num = blank_num

    blank = ''
    string_blank = ''
    for i in range(blank_num):
        blank += ' '
    for i in range(string_length):
        string_blank += ' '

    wall = ''
    for i in range((blank_num + string_length) * row_num + 2 + blank_num):
        wall += '='

    print('\n\n< model list >')
    print(wall)

    line = '|' + blank
    for i in range(column_num):
        for j in range(row_num):
            line += f'%{string_length}s' %(file_list[i*row_num + j]) + blank
            if j + 1 == row_num:
                line += '|'
            # line += f'{file_list[i*row_num + j]}    '
        print(f'{line}')
        line = '|' + blank

    line = '|' + blank
    if extra != 0:
        for i in range(extra):
            line += f'%{string_length}s' %(file_list[column_num*row_num + i]) + blank
            if i + 1 == extra:
                line += ((string_blank + blank) * (row_num - extra)) + '|'
        print(f'{line}')
    print(wall)


def pairwise_distance(point_cloud):
    if point_cloud.size()[0] == 1:
        point_cloud = torch.unsqueeze(torch.squeeze(point_cloud), dim=0)
    point_cloud_trans = torch.transpose(point_cloud, 1, 2)
    multi_mat = torch.mul(torch.matmul(point_cloud, point_cloud_trans), -2)
    point_cloud_square = torch.square(point_cloud)
    point_cloud_square_sum = torch.unsqueeze(torch.sum(point_cloud_square, 2), 0)
    point_cloud_square_sum_trans = torch.transpose(point_cloud_square_sum, 1, 2)
    Pairwise_matrix = -(point_cloud_square_sum + multi_mat + point_cloud_square_sum_trans)

    return Pairwise_matrix


def get_edge_feature(point_cloud, k_indices, k_value):
    if point_cloud.size()[0] == 1:
        point_cloud = torch.unsqueeze(torch.squeeze(point_cloud), dim=0)

    point_cloud_expand_copy = torch.unsqueeze(point_cloud, dim=2).repeat(1, 1, k_value, 1)

    feature_size = point_cloud.size()[-1]
    shape1 = k_indices.size()[1]
    shape2 = k_indices.size()[2]

    k_indices_flat = torch.reshape(k_indices, shape=[shape1 * shape2])
    knn_whole_flat = torch.index_select(point_cloud, dim=1, index=k_indices_flat)
    knn_whole = torch.reshape(knn_whole_flat, shape=[1, shape1, shape2, feature_size])


    edge_feature = torch.cat((point_cloud_expand_copy, knn_whole - point_cloud_expand_copy), dim=-1)
    return edge_feature


