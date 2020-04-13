import torch
import numpy as np
import os

model_path = '/home/011505052/thesis/Pseudo_Lidar_V2/results/sdn_argo_train/ckpt/'

model_list = os.listdir(model_path)

for idx, model in enumerate(model_list):
    checkpoint = torch.load(model)
    best_RMSE = checkpoint['best_RMSE']
    print(f'{idx}th checkpoint ... name {model}')
    print(best_RMSE)
