import numpy as np
import functions_my_1_2_1 as fmy
import copy
import random


class DataInformation:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data_vertices_list = []
        self.num_of_data = 0
        self.num_of_data_points = []
        self.up_sam_status = None
        self.rand_sam_status = None
        self.trans_status = None
        self.rot_status = None

    def __str__(self):
        return f'<{self.data_name}>\n' \
               f'number of data: {self.num_of_data}\n' \
               f'number of data points: {self.num_of_data_points}\n' \
               f'array size: {np.array(self.data_vertices_list).shape}\n' \
               f'up sample: {self.up_sam_status}\n' \
               f'random sample: {self.rand_sam_status}\n' \
               f'translate: {self.trans_status}\n' \
               f'rotation: {self.rot_status}'

    def up_sampling(self, up_num_of_stl_points):
        """
        :param up_num_of_stl_points:
        :return:
        """
        for data_index in range(self.num_of_data):
            temp_vertices_list = copy.deepcopy(self.data_vertices_list[data_index])
            print(f'data{data_index}: {np.array(temp_vertices_list).shape} --->', end=' ')
            for _ in range(up_num_of_stl_points - self.num_of_data_points[data_index]):
                rand_vertex = temp_vertices_list.pop(random.randint(0, len(temp_vertices_list) - 1))
                distance_of_rand_vertex_to_rest_vertices = fmy.euclidean_distance(rand_vertex, temp_vertices_list)
                nearest_vertex = temp_vertices_list[np.argmin(distance_of_rand_vertex_to_rest_vertices)]
                middle_vertex = list(np.average((np.array(rand_vertex), np.array(nearest_vertex)), axis=0))
                self.data_vertices_list[data_index].append(middle_vertex)
            print(np.array(self.data_vertices_list[data_index]).shape)
        self.num_of_data_points = up_num_of_stl_points
        self.up_sam_status = up_num_of_stl_points
        print()

    def random_sampling(self, up_num_of_stl_points, num_of_rand_sam):
        """
        :param up_num_of_stl_points:
        :param num_of_rand_sam:
        :return:
        """
        print(f'<{self.data_name}>')
        ran_sam_vertices_list = []
        for data_index in range(self.num_of_data):
            print(f'data{data_index}: {np.array(self.data_vertices_list[data_index]).shape} --->', end=' ')
            for _ in range(num_of_rand_sam):
                temp_list = []
                for _ in range(int(up_num_of_stl_points / num_of_rand_sam)):
                    rand_index = random.randint(0, len(self.data_vertices_list[data_index]) - 1)
                    temp_list.append(self.data_vertices_list[data_index].pop(rand_index))
                ran_sam_vertices_list.append(temp_list)
            print(f'{np.array(temp_list).shape} X {num_of_rand_sam}')
        self.data_vertices_list = ran_sam_vertices_list
        self.num_of_data *= num_of_rand_sam
        self.num_of_data_points = int(up_num_of_stl_points / num_of_rand_sam)
        self.rand_sam_status = f'rand/{num_of_rand_sam}'
        print()

    def translate(self, xyz_offset, num_of_trans):
        """
        :param xyz_offset:
        :param num_of_trans:
        :return:
        """
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        xyz_offset = np.expand_dims(np.array(xyz_offset), axis=-2)
        temp_list = [np.add(self.data_vertices_list[i], xyz_offset[i]).tolist() for i in range(self.num_of_data)]
        result_vertices = np.concatenate(temp_list, axis=0).tolist()
        print(f'{np.array(result_vertices).shape}')
        print()

        self.num_of_data *= num_of_trans
        self.data_vertices_list = result_vertices
        self.trans_status = f'transX{num_of_trans}'

    def rotation(self, num_of_rot, xyz_rot_matrices):
        """
        :param xyz_rot_matrices:
        :return:
        """
        print(f'<{self.data_name}>')
        print(f'data: {np.array(self.data_vertices_list).shape} --->', end=' ')
        center_vertex = np.average(np.array(self.data_vertices_list), axis=1)
        temp_list = []
        for i in range(self.num_of_data):
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

        self.num_of_data *= num_of_rot
        self.data_vertices_list = all_result_vertices.tolist()
        self.rot_status = f'rotX{num_of_rot}'

