import numpy as np


# num_sur_points = np.load(path + f'/num_skin_points.npy')
# num_B1_points = np.load(path + f'/num_skin_points.npy')
# num_B2_points = np.load(path + f'/num_skin_points.npy')

start_number = 52
while True:
    print(start_number)
    none_label = [35, 36, 37, 50, 69, 70, 85, 87]

    if start_number == 35 or start_number == 36 or start_number == 37 or start_number == 50 or start_number == 69 or start_number == 70 or start_number == 85 or start_number == 87:
        path = f'./data/data{start_number}'
    else:
        path = f'./data/data{start_number}(Label)'
    num_skin_points = np.load(path + f'/number_of_Skin.npy')
    num_bone1_points = np.load(path + f'/number_of_bone1.npy')
    num_bone2_points = np.load(path + f'/number_of_bone2.npy')
    # print(num_skin_points)
    # print(num_bone1_points)
    # print(num_bone2_points)

    np.save(path + f'/num_skin_points', num_skin_points)
    np.save(path + f'/num_bone1_points', num_bone1_points)
    np.save(path + f'/num_bone2_points', num_bone2_points)

    start_number += 1