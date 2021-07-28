import numpy as np

# Normalize function
def normalize_data(data):
    data_norm = []
    for array in data:
        element = np.array(array)
        min_val = np.min(array)
        element -= min_val

        max_val = np.max(element)
        element = element / max_val
        data_norm.append(element)

    data_norm = np.array(data_norm)

    return data_norm

data = np.array(
    [
        [
            [1, 2, 3, 6],
            [4, 7, 9, 5],
            [5, 6, 4, 2],
            [5, 1, 8, 6],
            [18, 9, 7, 11],
            [3, 7, 4, 6]
        ],
        [
            [11, 13, 12, 14],
            [5, 8, 9, 7],
            [1, 2, 3, 4],
            [9, 6, 3, 4],
            [13, 9, 1, 6],
            [12, 11, 10, 8]
        ]
    ]
)
# data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 12, 13, 15, 18, 20, 21, 22, 29]])
# print(data.shape)
# a = normalize_data(data)
# print(a)
# print(a.shape)