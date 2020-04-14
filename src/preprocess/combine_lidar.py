import argparse
import os

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Regular LiDAR with Pseudo-LiDAR')
    parser.add_argument('--velo_dir', type=str,
                        default='~/Kitti/object/training/calib')
    parser.add_argument('--pseudo_dir', type=str,
                        default='~/Kitti/object/training/predicted_disparity')
    parser.add_argument('--save_dir', type=str,
                        default='~/Kitti/object/training/predicted_velodyne')
    args = parser.parse_args()

    assert os.path.isdir(args.velo_dir)
    assert os.path.isdir(args.pseudo_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    velos = [x for x in os.listdir(args.velo_dir) if x[-3:] == 'bin']
    velos = sorted(velos)

    pseudos = [x for x in os.listdir(args.pseudo_dir) if x[-3:] == 'bin']
    pseudos = sorted(pseudos)

    for fn in velos:
        predix = fn[:-4]

        velo_file = '{}/{}.bin'.format(args.velo_dir, predix)
        velo = np.fromfile(velo_file, dtype=np.float32).reshape((-1,4))[:, :3]

        psuedo_file = '{}/{}.bin'.format(args.pseudo_dir, predix)
        psuedo = np.fromfile(psuedo_file, dtype=np.float32).reshape((-1,4))[:, :3]

        combined_lidar = np.vstack((velo,psuedo))
        combined_lidar = np.concatenate([combined_lidar, np.ones((combined_lidar.shape[0], 1))], 1)
        combined_lidar = combined_lidar.astype(np.float32)

        combined_lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish {}'.format(predix))