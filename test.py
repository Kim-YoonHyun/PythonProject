
import numpy as np
import copy
import random
import time
import dill
import math
#
# with open('input_data.pkl', 'rb') as f:
#     input_data = dill.load(f)
# print(input_data)


def make_rotate_angle(number, x1, x2, y1, y2, z1, z2):
    x_degree = random.uniform(x1, x2) * (math.pi / 180)
    y_degree = random.uniform(y1, y2) * (math.pi / 180)
    z_degree = random.uniform(z1, z2) * (math.pi / 180)
    return x_degree, y_degree, z_degree


xd, yd, zd = make_rotate_angle(2, -10, 10, -10, 10, -5, 5)

print(xd*180/math.pi)
print(yd*180/math.pi)
print(zd*180/math.pi)


def make_rotation_matrix(xyz_angle):
    x_matrix = np.expand_dims(np.array([
        [1.,                       0.,                      0.],
        [0.,   math.cos(xyz_angle[0]), -math.sin(xyz_angle[0])],
        [0.,   math.sin(xyz_angle[0]),  math.cos(xyz_angle[0])]]), axis=0)
    y_matrix = np.expand_dims(np.array([
        [math.cos(xyz_angle[1]),   0.,  math.sin(xyz_angle[1])],
        [0.,                       1.,                      0.],
        [-math.sin(xyz_angle[1]),  0.,  math.cos(xyz_angle[1])]]), axis=0)
    z_matrix = np.expand_dims(np.array([
        [math.cos(xyz_angle[2]), -math.sin(xyz_angle[2]),   0.],
        [math.sin(xyz_angle[2]),  math.cos(xyz_angle[2]),   0.],
        [0.,                       0.,                      1.]]), axis=0)
    return np.concatenate((x_matrix, y_matrix, z_matrix), axis=0)


x = make_rotation_matrix([xd, yd, zd])
b = np.random.randint(-6, 6, 36).reshape(6, 2, 3)

print(np.dot(b, x[0]))
print(np.dot(b[0,0], x[0]))
print(np.dot(b[0,1], x[0]))
print(np.dot(b[1,0], x[0]))
print(np.dot(b[1,1], x[0]))
print(np.dot(b[0,0], x[0]))