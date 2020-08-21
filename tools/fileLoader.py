# load depth pics or pcd files
import numpy as np
import cv2 as cv
from det3d.core.input.voxel_generator import VoxelGenerator
import glob
import pcl


def depth2points(depthImg, fx, fy, cx, cy, maxDepth):
    z = depthImg / 65536 * maxDepth
    xs = np.arange(z.shape[1])
    ys = np.arange(z.shape[0])
    xs = (xs - cx) / fx
    ys = (ys - cy) / fy
    X, Y = np.meshgrid(xs, ys)
    points = np.transpose((np.array([X * z, Y * z, z])), (1, 2, 0))
    return points


def getDataFromDepth(path, cfg):
    '''
    get point cloud data from depth image and calibration file
    and prepare the data for inputs
        path: depth image path
        calibPath: camera intrinsic path
    '''
    fs = cv.FileStorage(cfg.intrinsic, cv.FILE_STORAGE_READ)
    cameraMat = fs.getNode('CameraMat').mat()
    # distCoeff = fs.getNode('DistCoeff').mat()
    fs.release()
    fx = cameraMat[0, 0]
    fy = cameraMat[1, 1]
    cx = cameraMat[0, 2]
    cy = cameraMat[1, 2]

    points = depth2points(cv.imread(path, -1), fx, fy, cx, cy,
                          cfg.maxDepth).reshape((-1, 3))

    points = np.matmul(cfg.rotMat, points.T).T

    vgcfg = cfg.voxel_generator

    # create voxel generator
    vg = VoxelGenerator(
        voxel_size=vgcfg.voxel_size,
        point_cloud_range=vgcfg.range,
        max_num_points=vgcfg.max_points_in_voxel,
        max_voxels=vgcfg.max_voxel_num,
    )

    grid_size = vg.grid_size

    voxels, coordinates, num_points = vg.generate(points)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    data = dict(points=points,
                voxels=voxels,
                shape=grid_size,
                num_points=num_points,
                num_voxels=num_voxels,
                coordinates=coordinates)
    return data


def getBatchFromDepth(cfg):
    for path in glob.glob(cfg.depthPath + '*'):
        yield getDataFromDepth(path, cfg)


def getDataFromPcd(path, cfg):
    points = pcl.load(path).to_array()

    points = np.matmul(cfg.rotMat, points.T).T

    vgcfg = cfg.voxel_generator

    # create voxel generator
    vg = VoxelGenerator(
        voxel_size=vgcfg.voxel_size,
        point_cloud_range=vgcfg.range,
        max_num_points=vgcfg.max_points_in_voxel,
        max_voxels=vgcfg.max_voxel_num,
    )

    grid_size = vg.grid_size

    voxels, coordinates, num_points = vg.generate(points)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    data = dict(points=points,
                voxels=voxels,
                shape=grid_size,
                num_points=num_points,
                num_voxels=num_voxels,
                coordinates=coordinates)
    return data


def getBatchFromPcd(cfg):
    for path in glob.glob(cfg.depthPath + '*'):
        yield getDataFromPcd(path, cfg)