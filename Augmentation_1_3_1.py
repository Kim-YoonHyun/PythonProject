# _1_3  
# 함수 반복 실행시 기록이 되도록 status 에 += 으로 기록 계속 추가식으로 변경
# num_of_data, num_of_data_point 인스턴스 변수 삭제
# _1_3_1
# self.array_size 삭제 및 자체 계산 방식으로 변경

import numpy as np
import functions_my_1_3 as fmy
import copy
import random


class DataInformation:
    num = 0

    def __init__(self, data_name):
        self.data_name = data_name
        self.data_vertices_list = []
        self.up_sam_status = ''
        self.rand_sam_status = ''
        self.trans_status = ''
        self.rot_status = ''

        DataInformation.num += 1

    def __str__(self):
        return f'<{self.data_name}>\n' \
               f'array size: {np.array(self.data_vertices_list).shape}\n' \
               f'up sample:{self.up_sam_status}\n' \
               f'random sample:{self.rand_sam_status}\n' \
               f'translate:{self.trans_status}\n' \
               f'rotation:{self.rot_status}'

    def up_sampling(self, up_num_of_stl_points):
        """
        데이터 vertices의 갯수를 지정한 갯수만큼 추가 생성하는 함수
        :param up_num_of_stl_points: 추가생성을 통해 달성할 목표 갯수
        :return:
        """
        for data_index, data in enumerate(self.data_vertices_list):
            print(f'data{data_index}: {np.array(data).shape} --->', end=' ')
            # 추출용 copy vertices 생성
            vertices_list_copy = copy.deepcopy(data)
            num_of_data_points = np.array(data).shape[0]
            for _ in range(up_num_of_stl_points - num_of_data_points):
                # 랜덤으로 vertex 하나 추출
                rand_vertex = vertices_list_copy.pop(random.randint(0, len(vertices_list_copy) - 1))
                # 추출된 vertex 와 나머지 모든 vertex 간의 거리 계산
                distance_of_rand_vertex_to_rest_vertices = fmy.euclidean_distance(rand_vertex, vertices_list_copy)
                # 추출된 vertex 와 가장 가까운 vertex 선택
                nearest_vertex = vertices_list_copy[np.argmin(distance_of_rand_vertex_to_rest_vertices)]
                # 추출된 vertex 와 가장 가까운 vertex 의 중간지점에 새로운 vertex 생성
                middle_vertex = list(np.average((np.array(rand_vertex), np.array(nearest_vertex)), axis=0))
                # 생성된 새로운 vertex 를 기존 vertices 에 추가
                data.append(middle_vertex)
            print(np.array(data).shape)
        print()
        self.up_sam_status = up_num_of_stl_points

    def random_sampling(self, up_num_of_stl_points, num_of_rand_sam):
        """
        :param up_num_of_stl_points:
        :param num_of_rand_sam:
        :return:
        """
        print(f'<{self.data_name}>')
        ran_sam_vertices_list = []
        for data_index, data in enumerate(self.data_vertices_list):
            print(f'data{data_index}: {np.array(data).shape} --->', end=' ')
            for _ in range(num_of_rand_sam):
                temp_list = []
                for _ in range(int(up_num_of_stl_points / num_of_rand_sam)):
                    # 랜덤으로 vertex 하나 선택 후 추출
                    rand_index = random.randint(0, len(data) - 1)
                    temp_list.append(data.pop(rand_index))
                # 목표치 만큼 추출된 vertices list 를 저장
                ran_sam_vertices_list.append(temp_list)
            print(f'{np.array(temp_list).shape} X {num_of_rand_sam}')
        print()
        self.data_vertices_list = ran_sam_vertices_list
        self.rand_sam_status += f' rand/{num_of_rand_sam}'

    def translate(self, xyz_offset, num_of_trans):
        """
        :param xyz_offset:
        :param num_of_trans:
        :return:
        """
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        num_of_data = np.array(self.data_vertices_list).shape[0]
        xyz_offset = np.expand_dims(np.array(xyz_offset), axis=-2)
        temp_list = []
        for i in range(num_of_data):
            temp_list.append(np.add(self.data_vertices_list[i], xyz_offset[i]).tolist())

        temp_list = [np.add(self.data_vertices_list[i], xyz_offset[i]).tolist() for i in range(num_of_data)]
        result_vertices = np.concatenate(temp_list, axis=0).tolist()
        print(f'{np.array(result_vertices).shape}')
        print()
        self.data_vertices_list = result_vertices
        self.trans_status += f' transX{num_of_trans}'

    def rotation(self, num_of_rot, xyz_rot_matrices):
        """
        :param xyz_rot_matrices:
        :return:
        """

        num_of_data = np.array(self.data_vertices_list).shape[0]
        center_vertex = np.average(np.array(self.data_vertices_list), axis=1)
        temp_list = []
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        for i in range(num_of_data):
            # 원점으로 옮기기
            trans_to_zero_coordinate = np.subtract(np.array(self.data_vertices_list)[i], center_vertex[i])

            # rotation
            rot_vertices = np.dot(trans_to_zero_coordinate, xyz_rot_matrices[i])
            rot_vertices_expanded = np.expand_dims(rot_vertices, axis=-2)
            result_vertices_zero = np.concatenate((rot_vertices_expanded), axis=1)
            
            # 원래 위치로 되돌리기
            trans_to_origin_coordinate = np.add(result_vertices_zero, center_vertex[i])
            
            # list 에 넣기
            temp_list.append(trans_to_origin_coordinate)

        all_result_vertices = np.concatenate(temp_list, axis=0)
        print(f'{all_result_vertices.shape}')
        print()
        self.data_vertices_list = all_result_vertices.tolist()
        self.rot_status += f' rotX{num_of_rot}'