import os
import sys
import argparse
from PIL import Image
import numpy as np
import time

parser = argparse.ArgumentParser(description='apply masks on disparity map')
parser.add_argument('--depth_folder', type=str, default='', help='path to the disparity map folder', required=True)
parser.add_argument('--seg_mask_folder', type=str, default='', help='path to the segmentation mask', required=True)
parser.add_argument('--save_dir', type=str, default='', help='path to save your results', required=True)
args = parser.parse_args()

depth_dir = args.depth_folder
mask_dir = args.seg_mask_folder
depths = os.listdir(depth_dir)
masks = os.listdir(mask_dir)

if(len(depths) == len(masks)):
    print(f'There are {len(depths)} images to be processed.')

depths.sort()
masks.sort()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for depth_id, depth_name in enumerate(depths):
    # load disparity map (.npy)
    start_time_iter = time.time()
    depth_path = os.path.join(depth_dir, depth_name)
    depth = np.load(depth_path)
    
    # load corresponding mask
    mask_path = os.path.join(mask_dir, masks[depth_id])
    mask = Image.open(mask_path)
    mask = np.array(mask).astype(float) # turn Image -> numpy array
    mask[mask > 0.0] = 1.0 # turn anything greater than 0 to 1

    masked_disp = np.multiply(mask, depth) # apply masking
    save_path = args.save_dir + str(depth_id).zfill(6) + '.npy'
    np.save(save_path, masked_disp)
    end_time_iter = time.time()
    print(f'Inference takes {end_time_iter - start_time_iter} seconds')

    print('%04d/%04d: Masking done.' % (depth_id + 1, len(depths)))

print('Results saved.')