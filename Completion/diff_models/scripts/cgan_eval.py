import argparse
import os
import os.path as osp
import sys

sys.path.insert(-1, os.path.abspath(os.path.join(__file__, "..", "..")))

import math
import torch
import numpy as np
import trimesh
import json
from glob import glob
from einops import repeat, rearrange
from p_tqdm import p_map
from tqdm import tqdm

from utils.eval_utils import completeness, directed_hausdorff
from utils.chamfer import compute_trimesh_chamfer

SHAPENET_BASE = "/mnt/kostas-graid/datasets/ShapeNetReduced"

CGAN_EVAL_CATS = {
    "airplane": "02691156",
    "chair": "03001627",
    "table": "04379243",
}


def uhd_mmd_eval(model_glob, pcs_in_path):
    try:
        pcs_in = np.loadtxt(pcs_in_path)
        model_dirs = glob(model_glob)

        pcs_preds = list()
        for model_dir in model_dirs:
            if model_dir.endswith(".pts.npy"):
                sampled_pts = np.load(model_dir)
            elif model_dir.endswith(".obj"):
                mesh = trimesh.load(model_dir)
                sampled_pts, _ = trimesh.sample.sample_surface(mesh, 2048)
                np.savetxt(f"/home/wjhliang/temp/{os.path.split(model_dir)[-1]}.txt", sampled_pts)
            elif model_dir.endswith(".pts.txt"):
                sampled_pts = np.loadtxt(model_dir)
            else:
                raise ValueError(f"Unsupported file postfix: {model_dir}")
            pcs_preds.append(sampled_pts)

        # Completeness percentage
        gen_comp = 0
        for sample_pts in pcs_preds:
            comp = completeness(pcs_in, sample_pts)
            gen_comp += comp
        gen_comp = gen_comp / len(pcs_preds)

        # UHD
        pcs_preds_uhd = torch.from_numpy(np.array(pcs_preds))
        pcs_preds_uhd = rearrange(pcs_preds_uhd, "s n d -> s d n", d=3)

        pcs_in_uhd = torch.tensor(repeat(pcs_in, "n d -> s d n", s=10, d=3))

        hausdorff = directed_hausdorff(pcs_in_uhd, pcs_preds_uhd, reduce_mean=True).item()

        # TMD
        gen_pcs = pcs_preds
        sum_dist = 0
        for j in range(len(gen_pcs)):
            for k in range(j + 1, len(gen_pcs), 1):
                pc1 = gen_pcs[j]
                pc2 = gen_pcs[k]
                chamfer_dist = compute_trimesh_chamfer(pc1, pc2)
                sum_dist += chamfer_dist
        mean_dist = sum_dist * 2 / (len(gen_pcs) - 1)
    except Exception as e:
        # raise e
        print(e)
        mean_dist, hausdorff, gen_comp = np.nan, np.nan, np.nan

    return mean_dist, hausdorff, gen_comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument(
        "-c", "--category", type=str, help="chair, airplane, or table", default="chair"
    )
    parser.add_argument("-p", "--process", type=int, default=8)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-s", "--split", type=str, default="onet", choices=["onet", "cgan", "disn"])
    parser.add_argument("--dist_id", type=int, default=0)
    parser.add_argument("--dist_size", type=int, default=1)
    args = parser.parse_args()

    if args.category == "cgan_all":
        categories = CGAN_EVAL_CATS.keys()
    else:
        categories = [args.category]

    if osp.exists(osp.join(args.src, "points")):
        args.src = os.path.join(args.src, "points")
        print("Using pre-sampled 2048 points for evaluation")
    else:
        args.src = os.path.join(args.src, "meshes")  # For backward compatibiltiy

    for category in tqdm(categories):
        args.output = args.src.rstrip("/") + f"-eval_{category}_{args.dist_id}.txt"


        test_cat_id = CGAN_EVAL_CATS[category]
        if args.split == "onet":
            split_prefix = ""
        elif args.split == "disn":
            split_prefix = "disn_"
        else:
            split_prefix = "cgan_"
        with open(f"{SHAPENET_BASE}/{test_cat_id}/{split_prefix}test.lst", "r") as f:
            test_models = sorted(f.read().strip("\n").split("\n"))

        chunk_size = int(math.ceil(len(test_models) / args.dist_size))
        chunked_models = test_models[args.dist_id * chunk_size : (args.dist_id + 1) * chunk_size]

        if osp.basename(args.src) == "points":
            model_globs = [f"{args.src}/{test_cat_id}/{m_id}_*.pts.npy" for m_id in chunked_models]
            pcs_in_paths = [f"{args.src}/{test_cat_id}/{m_id}_input.txt" for m_id in chunked_models]
        else:
            model_globs = [f"{args.src}/{test_cat_id}/{m_id}_*.obj" for m_id in chunked_models]
            pcs_in_paths = [f"{args.src}/{test_cat_id}/{m_id}_input.txt" for m_id in chunked_models]

        tmds, uhds, comps = zip(
            *p_map(uhd_mmd_eval, model_globs, pcs_in_paths, num_cpus=args.process)
        )
        # tmds, uhds, comps = zip(*map(uhd_mmd_eval, model_globs, pcs_in_paths))  # For debugging

        with open(f"{args.src.rstrip('/')}-record_{category}_{args.dist_id}.txt", "w") as f:
            for m_id, tmd, uhd, comp in zip(chunked_models, tmds, uhds, comps):
                f.write(
                    f"ID: {m_id} \t TMD: {tmd:.4f} \t uhd: {uhd:.4f} \t completeness: {comp:.4f}\n"
                )

        with open(args.output, "w") as f:
            json.dump(
                {
                    "tmd": float(np.nanmean(tmds)),
                    "uhd": float(np.nanmean(uhds)),
                    "comp": float(np.nanmean(comps)),
                    "num_items": int(np.sum(~np.isnan(uhds) * ~np.isnan(tmds) * ~np.isnan(comps))),
                    "total_items": len(chunked_models),
                },
                f,
            )


if __name__ == "__main__":
    main()
