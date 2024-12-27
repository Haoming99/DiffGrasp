import argparse
# import numpy as np
import os
import sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)

from tqdm import tqdm
import pandas as pd
import trimesh
import torch
import math
from glob import glob
import numpy as np
import pickle


def summarize_results(generation_dir):
    eval_results = sorted(glob(os.path.join(generation_dir, 'eval_meshes_full_*.pkl')))
    out_file = os.path.join(generation_dir, f'eval_meshes_full.pkl')
    out_file_class = os.path.join(generation_dir, f'eval_meshes.csv')

    dfs = list()
    for fname in eval_results:
        with open(fname, 'rb') as f:
            dfs.append(pickle.load(f))

    eval_df = pd.concat(dfs)
    eval_df.to_pickle(out_file)

    meta_info_list = ['class id', 'class name', 'modelname']
    test_items = set(eval_df.columns.values.tolist()).difference(meta_info_list)
    eval_df_sum = eval_df[meta_info_list].copy()

    for item in test_items:
        eval_df_sum[f'{item} mean'] = eval_df[item].map(np.mean)
        eval_df_sum[f'{item} mode'] = eval_df[item].map(lambda x: x[0] if isinstance(x, list) else np.nan) # The first item was generateed with z_0
        eval_df_sum[f'{item} min'] = eval_df[item].map(np.min)
        eval_df_sum[f'{item} max'] = eval_df[item].map(np.max)

    # Create CSV file  with main statistics
    eval_df_class = eval_df_sum.groupby(by=['class name']).mean()
    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()

    eval_df_class.to_csv(out_file_class)

    pd.options.display.max_columns = len(eval_df_class.columns)
    print(eval_df_class)

    fnames = glob(f"{generation_dir}/vis/*/*.off")
    for fname in tqdm(fnames, desc='Transforming visulizations'):
        obj_name = fname.rstrip('.off') + '.obj'
        mesh = trimesh.load(fname, process=False)
        _ = mesh.export(obj_name)
        os.system(f"rm {fname}")
    print("Done!")


if __name__ == '__main__':
    # cfg = config.load_config(sys.argv[1], 'configs/default.yaml')
    # out_dir = cfg['training']['out_dir']
    # generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    summarize_results(sys.argv[1])

