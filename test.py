
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd

# s = pd.Series([100, 200, 300])
# # print(s, '\n')
#
# s = pd.Series([100, 200, 300],
#               index=['서울', '대구', '부산'], dtype=np.int32)
#
# # print(s)
# # print(type(s.values))
# # print(type(s.index))
# # print(s.index)
# s.name = '인구'
# s.index.name = '도시'
# # print(s)
# # print(s/10)
# # print(s[0])
# # print(s['서울'])
# # print(s.서울)
# # print(s[[0, 2]])
# # print(s[['서울', '부산']])
# # print(s[1:3])
# # print(s['대구':'부산'])
# # print(s[(100 < s) & (s < 300)])
# # print(s[(s < 100) | (s > 200)])
# s2 = pd.Series({'서울':300, '대구':200, '대전':100})
# rs = s - s2
# # print((s-s2)/s * 100)
# # print(s.values - s2.values)
# # print(rs.notnull())
# # print(rs[rs.notnull()])
# rs = rs[rs.notnull()]
# rs['청주'] = 600
# print(rs)
# del rs['서울']
# print(rs)

data = {
    "지역": ["수도권", "경상권", "경상권"],
    "2015": [100, 200, 300],
    "2010": [400, 500, 600],
    "2005": [700, 800, 900],
    "2000": [1000, 1100, 1200],
    "2010-2015 증가율": [0.01, 0.02, 0.03]
}
columns = ["지역", "2000", "2005", "2010", "2015", "2010-2015 증가율"]
index = ["서울", "부산", "대구"]
df = pd.DataFrame(data, index=index, columns=columns)
# print(df)
# print(df.지역.values)
# print(df['2000'])
# print(df.columns)
# print(df.columns.values)
df.index.name = '도시'
df.columns.name = '특성'
# print(df.T)
df['2010-2015 증가율'] = df['2010-2015 증가율'] * 100
# print(df)
df['2020'] = [1, 2, 3]
# print(df
del df['2020']
# print(df)
# print(df['지역'])
# print(df[['지역']])
# print(df[['지역', '2015']])
# print(df['서울':'부산'])
# print(df[1:3])
# print(df.head(2))
# print(df.head())
# print(df['2015']['서울'])
# print(df['서울']['2015'])
df['2015']['서울'] = 100
print(df.values)