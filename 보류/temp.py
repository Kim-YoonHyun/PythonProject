import numpy as np
labeled_data = np.load('./data/data33/data_labeled.npy')
print(labeled_data)
print(labeled_data.shape)
data = labeled_data[:, :, 0:3]

np.save('./data/data33/data.npy', data)