import numpy as np
import pandas as pd


a = np.arange(0, 12).reshape(2, 2, 3)
b = np.tile(a, [1, 1, 2])
print(a)
print(b)
c = np.concatenate(b, axis=0)
print(c)
print(a.shape)
print(b.shape)
print(c.shape)
