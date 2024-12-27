import argparse
import os
import os.path as osp
import sys
sys.path.insert(-1, os.path.abspath(os.path.join(__file__, '..', '..')))

import math
import torch
import numpy as np
# import trimesh
import json
from glob import glob
# from einops import repeat, rearrange
# from p_tqdm import p_map
from tqdm import tqdm
import pandas as pd

from utils.gca_metrics import (
    mutual_difference,
    unidirected_hausdorff_distance,
    MMDCalculator,
    compute_chamfer_l1
)

class PredPcsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, out_dir):
        models = list()
        split = 'test'
        for room_path in sorted(glob(f"{dataset_folder}/room*/")):
            room_name = osp.basename(osp.dirname(room_path))
            with open(f"{room_path}/{split}.lst", 'r') as f:
                scene_ids = f.read().splitlines()
                models.extend([{'room_name': room_name, 'id': scene_id} for scene_id in scene_ids])
        self.models = models
        self.out_dir = out_dir
        self.dataset_folder = dataset_folder
    
    def __getitem__(self, index):

        model = self.models[index]
        
        try:
            pred_pcs_paths = glob(f"{self.out_dir}/points/{model['room_name']}/{model['id']}_[0-9]*.pts.npy")
            pred_pcs = np.array([np.load(m) for m in pred_pcs_paths])
            pred_pcs = torch.from_numpy(pred_pcs).float()
            
            gt_pcs = dict(np.load(f"{self.dataset_folder}/test/{model['room_name']}/{model['id']}/surface.npz"))['surface']
            gt_pcs = torch.from_numpy(gt_pcs).float()
            
            in_pcs = torch.tensor(np.loadtxt(f"{self.out_dir}/points/{model['room_name']}/{model['id']}_input.txt")).float()
            data_dict = {'room_name': model['room_name'], 'id': model['id'], 'pred_pcs': pred_pcs, 'gt_pcs': gt_pcs, 'in_pcs': in_pcs}

        except Exception as e:
            print(f"Failed to load model {model}")
            print(e)
            data_dict = {'room_name': model['room_name'], 'id': model['id'], 
                'pred_pcs': torch.empty((0, 3)), 'gt_pcs': torch.empty((0, 3)), 'in_pcs': torch.empty((0, 3))}
        
        return data_dict
        
    def __len__(self):
        return len(self.models)


def scene_eval(pred_pcs, gt_pcs, in_pcs):
    try:
        chamfer_l1 = compute_chamfer_l1(pred_pcs, gt_pcs)
        chamfer_l1 = np.array(chamfer_l1)
        min_chamfer_l1 = chamfer_l1.min()
        mean_chamfer_l1 = chamfer_l1.mean()

        tmd = mutual_difference(pred_pcs)
        uhd = unidirected_hausdorff_distance(in_pcs, pred_pcs)

    except Exception as e:
        # raise e
        print(e)
        # breakpoint()
        chamfer_l1, min_chamfer_l1, mean_chamfer_l1, tmd, uhd = np.array([]), np.nan, np.nan, np.nan, np.nan

    return {'chamfer_l1': chamfer_l1.tolist(), 'min_chamfer_l1': min_chamfer_l1, 'mean_chamfer_l1': mean_chamfer_l1, 'tmd': tmd, 'uhd': uhd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("-p", "--process", type=int, default=8)
    parser.add_argument("-o", "--output", type=str, default='')
    parser.add_argument("-d", "--dataset", type=str, default='/scratch/wen/synthetic_room_dataset/')
    parser.add_argument("--dist_id", type=int, default=0)
    parser.add_argument("--dist_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda')
    dataset = PredPcsDataset(args.dataset, args.src)

    chunk_size = int(math.ceil(len(dataset) / args.dist_size))
    chunked_dataset = torch.utils.data.Subset(dataset, range(args.dist_id * chunk_size, min((args.dist_id + 1) * chunk_size, len(dataset))))
    pred_loader = torch.utils.data.DataLoader(chunked_dataset, batch_size=1, num_workers=8, drop_last=False, shuffle=False)

    if args.output == '':
        args.output = osp.join(args.src, f"eval_{args.dist_id}.json")
    
    eval_records = {'min_chamfer_l1': list(), 'mean_chamfer_l1': list(), 'tmd': list(), 'uhd': list(), 
                    'chamfer_l1': list(), 'room_name': list(), 'id': list()}
    print("Starting evaluation...")

    for batch in tqdm(pred_loader):
        pred_pcs, gt_pcs, in_pcs = batch['pred_pcs'][0].to(device), batch['gt_pcs'][0].to(device), batch['in_pcs'][0].to(device)
        result = scene_eval(pred_pcs, gt_pcs, in_pcs)
        

        for k, v in result.items():
            eval_records[k].append(v)    

        eval_records['room_name'].append(batch['room_name'][0])

        eval_records['id'].append(batch['id'][0])
   
    df = pd.DataFrame(data=eval_records)
    df.to_csv(osp.join(args.src, f"record_{args.dist_id}.csv"))

    uhds = np.array(eval_records['uhd'])
    tmds = np.array(eval_records['tmd'])
    mean_chamfer_l1 = np.array(eval_records['mean_chamfer_l1'])
    min_chamfer_l1 = np.array(eval_records['min_chamfer_l1'])

    with open(args.output, 'w') as f:
        json.dump({'tmd': float(np.nanmean(tmds)),
            'uhd': float(np.nanmean(uhds)),
            'mean_chamfer_l1': float(np.nanmean(mean_chamfer_l1)),
            'min_chamfer_l1': float(np.nanmean(min_chamfer_l1)),
            'num_items': int(np.sum(~np.isnan(uhds) * ~np.isnan(tmds) * ~np.isnan(mean_chamfer_l1) * ~np.isnan(min_chamfer_l1))),
            'total_items': len(chunked_dataset)
            }, f)



if __name__ == '__main__':
    main()
