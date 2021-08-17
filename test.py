
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd

num_of_rot = 3  # rotation 을 통해 증가시킬 배수
rot_x = [-5.0, 5.0]  # x 축에 대한 최대 rotation 범위
rot_y = [-5.0, 5.0]  # y 축에 대한 최대 rotation 범위
rot_z = [-5.0, 5.0]  # z 축에 대한 최대 rotation 범위


all_rot_matrix = []
num_of_data = 2
for j in range(num_of_data):
    xyz_rot_matrix = []
    for i in range(num_of_rot):
        x_angle = random.uniform(rot_x[0], rot_x[1]) * (math.pi / 180)
        y_angle = random.uniform(rot_y[0], rot_y[1]) * (math.pi / 180)
        z_angle = random.uniform(rot_z[0], rot_z[1]) * (math.pi / 180)

        x_matrix = np.array([
                        [1.,                        0.,                         0.],
                        [0., math.cos(x_angle), -math.sin(x_angle)],
                        [0., math.sin(x_angle), math.cos(x_angle)]])
        y_matrix = np.array([
                        [ math.cos(y_angle), 0., math.sin(y_angle)],
                        [                        0., 1.,                        0.],
                        [-math.sin(y_angle), 0., math.cos(y_angle)]])
        z_matrix = np.array([
                        [math.cos(z_angle), -math.sin(z_angle), 0.],
                        [math.sin(z_angle),  math.cos(z_angle), 0.],
                        [                       0.,                         0., 1.]])
        xy_matrix = np.dot(x_matrix, y_matrix)
        xyz_matrix = np.dot(xy_matrix, z_matrix)
        xyz_rot_matrix.append(xyz_matrix)
    all_rot_matrix.append(xyz_rot_matrix)


# print(np.array(all_rot_matrix))
# print(np.array(all_rot_matrix).shape)
b = np.arange(0, num_of_data*2*3).reshape(num_of_data, 2, 3)
# print(b)
# print(b.shape)

temp_list = []
for j in range(num_of_data):
    # print(b[j])
    print(b[j].shape)
    # print(np.array(all_rot_matrix)[j])
    print(np.array(all_rot_matrix)[j].shape)
    c = np.dot(b[j], all_rot_matrix[j])
    print(c)
    print(c.shape)
    e = np.expand_dims(c, axis=-2)
    print(e.shape)

    ee = np.concatenate((e), axis=1)
    print(ee)
    print(ee.shape)
    temp_list.append(ee)
print(np.array(temp_list).shape)
temp_array = np.concatenate((temp_list), axis=0)
print(temp_array)
print(temp_array.shape)


