import os
import imageio
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(-1, os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
# import torch.backends.cudnn as cudnn
# import torchvision.utils as vutils
# cudnn.benchmark = True

from utils.fresnel_vis import gray_color, renderMeshCloud, gold_color
from einops import repeat, rearrange

import trimesh
import os.path as osp
import argparse
import torchvision
from tqdm import tqdm
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", type=str)
    parser.add_argument("--dist_id", type=int, default=0)
    parser.add_argument("--dist_size", type=int, default=1)



    args = parser.parse_args()
    # cgan_all = ['03001627', '02691156', '04379243']
    # cgan_all = ['03001627', '04379243'] # No airplane because lack of diversity
    # cgan_all = ['03001627', '02691156', '04379243']
    # cgan_all = ['03001627', '04379243'] # No airplane because lack of diversity
    cgan_all = ['02691156', '02828884',  '02933112',  '02958343',  '03001627',  '03211117',  '03636649',
        '03691459', '04090263', '04256520',  '04379243',  '04401088',  '04530566']
    job_id = os.environ['SLURM_JOB_ID']
    # auto_base = '/home/wen/paper_vis/autosdf/'

    exp_name = args.exp_path.rstrip('/').split('/')[-1]

    out_dir = f'/home/wen/paper_vis_reb/{exp_name}/preview-test/'
    task_list = []

    for cat_id in cgan_all:
        os.makedirs(f"{out_dir}/{cat_id}", exist_ok=True)
        split_file = f'/home/wen/ShapeNet/{cat_id}/cgan_val.lst'
        # split_file = f'/home/wen/ShapeNet/{cat_id}/cgan_test.lst'
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        task_list += [(cat_id, model_id) for model_id in models_c]

    chunk_size = int(math.ceil(len(task_list) / args.dist_size))
    chunked_tasks = task_list[args.dist_id * chunk_size: (args.dist_id + 1) * chunk_size]

    time_str = datetime.now().strftime("%m%d-%H:%M")
    name_str = f"{args.dist_id}-{args.dist_size}-{time_str}-{job_id}.start"
    status_fn = f"{out_dir}/../{name_str}"
    with open(status_fn, "w") as f:
        f.write(name_str)

    resolution = (128, 128)
    sample_times = 4
    vis_camera = dict(camPos=np.array([1, 1, 1]), camLookat=np.array([0., 0., 0.]),
                    camUp=np.array([0, 1, 0]), camHeight=1, 
                    resolution=resolution, samples=sample_times, light_samples=sample_times)



    for cat_id, model_id in tqdm(chunked_tasks, desc="Main Task"):
        try:
            cur_imgs = list()

            pcs_in = np.loadtxt(f"{args.exp_path}/meshes/{cat_id}/{model_id}_input.txt")
            res_img = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
            rendered_img = renderMeshCloud(cloud=pcs_in, cloudR=0.01, cloudC=gold_color, **vis_camera)
            res_img += rendered_img[:, :, :-1] * (rendered_img[:, :, None, -1] > 0) + (255 - rendered_img[..., None, -1])

            cur_imgs.append(res_img)


            mesh_fns = [f"{args.exp_path}/meshes/{cat_id}/{model_id}_{b_idx:02d}.obj" for b_idx in range(10)]

            for mesh_fn in mesh_fns:
                res_img = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)

                try:
                    cur_mesh = trimesh.load(mesh_fn)
                    xg_mesh = dict(vert=cur_mesh.vertices, face=cur_mesh.faces)

                    rendered_img = renderMeshCloud(mesh=xg_mesh, meshC=gray_color, **vis_camera)
                    res_img += rendered_img[:, :, :-1] * (rendered_img[:, :, None, -1] > 0) + (255 - rendered_img[..., None, -1])
                except Exception as e:
                    print(e)

                cur_imgs.append(res_img[:, ::-1])

            all_imgs = [torch.tensor(i.copy()).permute([2, 0, 1]) for i in cur_imgs]

            grid_img = torchvision.utils.make_grid(all_imgs, 3)
            grid_img = rearrange(grid_img, 'c h w -> h w c').numpy()

            plt.imsave(f"{out_dir}/{cat_id}/{model_id}.jpg", grid_img)

        except Exception as e:
            print(e)

    done_fn = status_fn.replace('.start', '.done')
    os.system(f"mv {status_fn} {done_fn}")

if __name__ == '__main__':
    main()
