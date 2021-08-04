import numpy as np
import copy
import random


a = np.array([[[1, 2, 3]]]).tolist()
# b = np.array([[[4, 5, 6]]]).tolist()
# c = np.array([[[7, 8, 9]]]).tolist()
# d = np.array([[[10, 11, 12]]]).tolist()
# e = np.concatenate((a, b, c, d), axis=0)
# print(e)
# print(e.shape)
# f = [a, b, c, d]
# print(f)
# g = np.concatenate(f, axis=0)
# print(g)
# print(g.shape)

b = np.array(a)
b = [b.shape[0], b.shape[1], b.shape[2]]

print(b)