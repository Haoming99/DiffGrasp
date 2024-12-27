"""
Evaluation following conv onet protocol
"""
import argparse
import os
import os.path as osp
import re
import sys
sys.path.insert(-1, os.path.abspath(os.path.join(__file__, '..', '..')))

import numpy as np
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
#from im2mesh import config, data
#from im2mesh.eval import MeshEvaluator
#from im2mesh.utils.io import load_pointcloud
from utils.onet_evaluator import MeshEvaluator
from utils.sn_pcs_dataset import ShapetNetTestSet
import math


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('generation_dir', type=str, help='Path to generated_meshes')
parser.add_argument('--shapenet_folder', type=str, default='/Datasets/ShapeNet/', help='Path to Shapnet data from onet')
parser.add_argument('--split', type=str, default='test', help='test split for debugging')
parser.add_argument('--num_samples', type=int, default=10, help='num_samples to test')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')
parser.add_argument('--dist_id', type=int, default=0, help='Run evaluation in parallel')
parser.add_argument('--dist_size', type=int, default=1, help='Run evaluation in parallel')

args = parser.parse_args()

# Shorthands
generation_dir = args.generation_dir # os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
    out_file = os.path.join(generation_dir, f'eval_meshes_full_{args.dist_id}.pkl')
    out_file_class = os.path.join(generation_dir, f'eval_meshes_{args.dist_id}.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

dataset = ShapetNetTestSet(args.shapenet_folder, split=args.split)
print('Test split: ', dataset.split)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
chunk_size = int(math.ceil(len(dataset) / args.dist_size))
chunked_dataset = torch.utils.data.Subset(dataset, range(args.dist_id * chunk_size, min((args.dist_id + 1) * chunk_size, len(dataset))))

test_loader = torch.utils.data.DataLoader(
    chunked_dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()

    modelname = data['model'][0]
    category_id = data['category'][0]

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, category_id)
        pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # Evaluate
    pointcloud_tgt = data['pointcloud_tgt'].squeeze(0).float().numpy()
    normals_tgt = data['normal_tgt'].squeeze(0).float().numpy()
    points_tgt = data['points_tgt'].squeeze(0).float().numpy()
    occ_tgt = data['occ_tgt'].squeeze(0).float().numpy()
    scale = data['scale'].squeeze(0).float().numpy()
    loc = data['loc'].squeeze(0).float().numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if args.num_samples >= 1: # New version
        mesh_files = [os.path.join(mesh_dir, '%s_%02d.obj' % (modelname, i)) for i in range(args.num_samples)]
    else: # num_samples == 0 means the original protocal of onet
        mesh_files = [os.path.join(mesh_dir, '%s.obj' % modelname)]

    for mesh_idx, mesh_file in enumerate(mesh_files):

        if os.path.exists(mesh_file):
            mesh = trimesh.load(mesh_file, process=False)
            mesh.vertices = mesh.vertices / scale - loc # Because the model has been denoramlized in generation
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
            for k, v in eval_dict_mesh.items():
                entry_name = k + f' (mesh)'
                if entry_name in eval_dict:
                    eval_dict[entry_name].append(v)
                else:
                    eval_dict[entry_name] = [v]
                #eval_dict[k + f' (mesh) {mesh_idx:02d}'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    ## Evaluate point cloud
    #if cfg['test']['eval_pointcloud']:
    #    pointcloud_file = os.path.join(
    #        pointcloud_dir, '%s.ply' % modelname)

    #    if os.path.exists(pointcloud_file):
    #        pointcloud = load_pointcloud(pointcloud_file)
    #        eval_dict_pcl = evaluator.eval_pointcloud(
    #            pointcloud, pointcloud_tgt)
    #        for k, v in eval_dict_pcl.items():
    #            eval_dict[k + ' (pcl)'] = v
    #    else:
    #        print('Warning: pointcloud does not exist: %s'
    #                % pointcloud_file)


#meta_info_set = {'idx', 'class id', 'class name', 'modelname'}
#pattern = re.compile(r' \d+$')
#df_head = {p.sub('', i, count=1) for i, p in zip(cols, itertools.repeat(pattern))}
#items_root = df_head.difference(meta_info_set)


# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

meta_info_list = ['class id', 'class name', 'modelname']
test_items = set(eval_df.columns.values.tolist()).difference(meta_info_list)
eval_df_sum = eval_df[meta_info_list].copy()

for item in test_items:
    eval_df_sum[f'{item} mean'] = eval_df[item].map(np.mean)
    eval_df_sum[f'{item} mode'] = eval_df[item].map(lambda x: x[0])
    eval_df_sum[f'{item} min'] = eval_df[item].map(np.min)
    eval_df_sum[f'{item} max'] = eval_df[item].map(np.max)

# Create CSV file  with main statistics
eval_df_class = eval_df_sum.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
