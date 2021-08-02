import vtk

# 입력한 point cloud 로 Actor 를 만드는 함수
def point_actor(point_cloud, point_size, color):
    color_preset = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [160, 160, 160],
        [255, 51, 204],
        [255, 255, 255],
        [90, 90, 90],
        [255, 100, 100],
        [0, 0, 0]
    ]
    output_points = vtk.vtkPoints()
    output_vertices = vtk.vtkCellArray()
    output_scalars = vtk.vtkFloatArray()
    output_scalars.SetNumberOfComponents(1)
    output_scalars.SetName("Sequence")

    output_colors = vtk.vtkUnsignedCharArray()
    output_colors.SetNumberOfComponents(3)
    output_colors.SetName("Colors")

    for point in point_cloud:
        id = output_points.InsertNextPoint(point[:3])
        output_vertices.InsertNextCell(1)
        output_vertices.InsertCellPoint(id)
        output_scalars.InsertNextTuple([1.0])
        if color == 'Red':
            output_colors.InsertNextTuple(color_preset[0])
        if color == 'Blue':
            output_colors.InsertNextTuple(color_preset[1])
        if color == 'Green':
            output_colors.InsertNextTuple(color_preset[2])
        if color == 'Yellow':
            output_colors.InsertNextTuple(color_preset[3])
        if color == 'Double':
            output_colors.InsertNextTuple(color_preset[int(point[3])])
        if color == 'Gray':
            output_colors.InsertNextTuple(color_preset[4])
        if color == 'Pink':
            output_colors.InsertNextTuple(color_preset[5])
        if color == 'White':
            output_colors.InsertNextTuple(color_preset[6])
        if color == 'Dark Gray':
            output_colors.InsertNextTuple(color_preset[7])
        if color == 'Light Red':
            output_colors.InsertNextTuple(color_preset[8])
        if color == 'Black':
            output_colors.InsertNextTuple(color_preset[9])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(output_points)  # Point data 를 입력
    polydata.SetVerts(output_vertices)  # vertex 정보를 입력
    if color == 'None':
        polydata.GetPointData().SetScalars(output_scalars)  # 위에서 지정한 색 입력
    else:
        polydata.GetPointData().SetScalars(output_colors)  # 위에서 지정한 색 입력
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)

    return actor

# value assign function
def assign_value_to_polydata(polydata, scalar_data):
    for idx, scalar in enumerate(scalar_data):
        polydata.GetPointData().GetScalars().SetTuple(idx, [scalar])

    polydata.GetPointData().Modified()