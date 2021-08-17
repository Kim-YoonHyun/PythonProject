import vtk
import numpy as np

def read_stl(path, stl_name):
    """
    function of read stl file vertices and number of point information

    path: stl file path
    stl_name: stl file name
    return: stl vertices list, number of stl coordinate points
    """
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(f'{path}/{stl_name}.stl')
    file_data.Update()
    num_of_stl_points = file_data.GetOutput().GetNumberOfPoints()
    stl_vertices_list = list(
        list(file_data.GetOutput().GetPoint(i)) for i in range(num_of_stl_points)
    )
    return stl_vertices_list, num_of_stl_points

def euclidean_distance(start, end):
    """
    function of calculate the euclidean distance from start to end
    :param start: start vertex
    :param end: end vertex
    :return: euclidean distance
    """

    value = np.sqrt(np.sum(np.square(np.subtract(start, end)), axis=-1))
    return value