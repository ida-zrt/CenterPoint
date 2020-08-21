# pipeline for img, depth reading
# load depth, img
# convert to point clouds
# call voxelization
# format data
# put it into the model and test
# still need to figure out how voxelization is done!!

import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual
from collections import defaultdict
import mayavi.mlab as mlab
import cv2 as cv

from tools.visualize import showPoints, createRotation
from tools.fileLoader import *


if __name__ == "__main__":
    cfg = Config.fromfile('configs/centerpoint/myconfig.py')
    for data in getBatchFromDepth(cfg):

        points = data['points']
        voxels = data['voxels']

        # static of all points
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        print('X: max {}, min {}'.format(xs.max(), xs.min()))
        print('Y: max {}, min {}'.format(ys.max(), ys.min()))
        print('Z: max {}, min {}'.format(zs.max(), zs.min()))

        # # DO NOT display them in the for loop!
        # # display all points
        # showPoints(points)

        # # display selected voxels
        # voxelPoints = np.concatenate(voxels.tolist())
        # showPoints(voxelPoints)

    # mlab.show()
