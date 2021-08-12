
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd


s = pd.Series([100, 200, 300], index=['인덱스0', '인덱스1', '인덱스2'], dtype=np.int32)
# print(s.index.values)
# print(s[0])
# print(s['인덱스0'])
# print(s.인덱스0)
# print(s[[0, 2]])
# print(s[['인덱스0', '인덱스2']])
# print(s[1:2])
# print(s['인덱스1':'인덱스2'])
s['인덱스3'] = 400
# s.인덱스4 = 400
# print(s)
# print(s.notnull())

data = {
    "열0": ['aa', 'bb', 'cc'],
    "열1": [100, 200, 300],
    "열2": [400, 500, 600],
    "열3": [700, 800, 900],
    "열4": [1000, 1100, 1200],
    "열5": [0.01, 0.02, 0.03]
}
columns = ["열0", "열1", "열2", "열3", "열4", "열5"]
index = ["인덱스0", "인덱스1", "인덱스2"]

df = pd.DataFrame(data, index=index, columns=columns)


# print(df)
# print(df.열0[0])
# print(df.열0['인덱스0'])
# print(df.열0.인덱스0)
# print(df.열0.values)
# print(df.columns.values)
df['열6'] = [1, 2, 3]
# print(df)
# print(df[[13]])
df.to_csv('sample.csv')
# df2 = pd.read_csv('sample.csv', names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'])
df2 = pd.read_csv('sample.csv', header=None)

data = [
    ['aa', 'bb', 'cc'],
    [100, 200, 300],
    [400, 500, 600]
]
index = ["인덱스0", "인덱스1", "인덱스2"]

df3 = pd.DataFrame(data, index=index)
# print(df3)
# print(df3[[0]])
# print(df.loc[['인덱스1']:['인덱스2']])
print(df[])

