import numpy as np

a = []

stl_vertices = list(
        list((-9.928400039672852, -9.393500328063965, 33.91360092163086)) for i in range(2)
    )
a.append(stl_vertices)
a.append(stl_vertices)
print(np.array(a))