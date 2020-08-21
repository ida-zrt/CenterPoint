import numpy as np
import mayavi.mlab
import pcl
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel

fname = './demo/nuScenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883533049931.pcd.bin'
pointcloud = np.fromfile(fname, dtype=np.float32).reshape(-1, 5)
# fname = './data/sjtuRoad/data_2020_07_29_17_44_03/pointCloud/No.0001.pcd'
# pointcloud = pcl.load(fname).to_array()

x = pointcloud[:, 0]
y = pointcloud[:, 1]
z = pointcloud[:, 2]

# r = pointcloud[:, 3]
col = z

fig = mayavi.mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(640, 500))

mayavi.mlab.points3d(
    x,
    y,
    z,
    col,  # Values used for Color
    mode="point",
    colormap='spectral',  # 'bone', 'copper', 'gnuplot'
    # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
    figure=fig)

mayavi.mlab.show()
