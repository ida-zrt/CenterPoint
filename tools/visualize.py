import mayavi.mlab as mlab
import numpy as np
import math


def showPoints(points):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    col = zs
    fig = mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(640, 500))
    mlab.points3d(xs, ys, zs, col, mode="point", figure=fig)


def createRotation(angles1):
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0] * 3.141592653589793 / 180.0
    theta[1] = angles1[1] * 3.141592653589793 / 180.0
    theta[2] = angles1[2] * 3.141592653589793 / 180.0
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
