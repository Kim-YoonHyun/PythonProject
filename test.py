
import numpy as np
import copy
import random
import time
import dill
import math
import pandas as pd
import functions_my

np.random.seed(2021)
a = np.random.randint(-15, 15, [3, 10])
print(a)
print(a.shape)
print(np.max(a, axis=1))
print(np.max(a, axis=1).shape)

