import argparse
import os
import sys
sys.path.insert(-1, os.path.abspath(os.path.join(__file__, '..', '..')))

import numpy as np
import math
import trimesh
from utils.chamfer import compute_trimesh_chamfer
from glob import glob
from joblib import Parallel, delayed
from p_tqdm import p_map


def process_one(shape_dir):
    pc_paths = glob(shape_dir)
    pc_paths = sorted(pc_paths)
    gen_pcs = []
    for path in pc_paths:
        loaded_obj = trimesh.load(path, process=False)
        sample_pts = trimesh.sample.sample_surface(loaded_obj, 2048) # sample 2048 points as MPC's implementation
        gen_pcs.append(sample_pts)

    sum_dist = 0
    for j in range(len(gen_pcs)):
        for k in range(j + 1, len(gen_pcs), 1):
            pc1 = gen_pcs[j]
            pc2 = gen_pcs[k]
            chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
            sum_dist += chamfer_dist
    mean_dist = sum_dist * 2 / (len(gen_pcs) - 1)
    return mean_dist


def Total_Mutual_Difference(args):
    objs = glob(f'{args.src}/*.obj')
    shape_names = sorted(list(set(os.path.basename(i).split('_')[0] for i in objs)))
    chunk_size = int(math.ceil(len(shape_names) / args.dist_size))
    chunked_names = shape_names[args.dist_id * chunk_size: min((args.dist_id + 1) * chunk_size, len(shape_names))]

    res = 0
    all_shape_dir = [f"{args.src}/{name}_[0-9]*.obj" for name in chunked_names]

    #results = Parallel(n_jobs=args.process, verbose=2)(delayed(process_one)(path) for path in all_shape_dir)
    results = p_map(process_one, all_shape_dir, num_cpus=args.process)

    info_path = args.src + f'-record_meandist_{args.dist_id}.txt'
    with open(info_path, 'w') as fp:
        for i in range(len(chunked_names)):
            print("ID: {} \t mean_dist: {:.4f}".format(chunked_names[i], results[i]), file=fp)
    res = np.mean(results)

    np.savetxt(args.output, [len(results), res])

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("-p", "--process", type=int, default=10)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--dist_id", type=int, default=0)
    parser.add_argument("--dist_size", type=int, default=1)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.src.rstrip('/') + f'-eval_TMD_{args.dist_id}.txt'

    res = Total_Mutual_Difference(args)
    print("Avg Total Multual Difference: {}".format(res))

    #with open(args.output, "w") as fp:
    #    fp.write("SRC: {}\n".format(args.src))
    #    fp.write("Total Multual Difference: {}\n".format(res))


if __name__ == '__main__':
    main()
