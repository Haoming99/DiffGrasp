import argparse
import numpy as np
import os
from tqdm import tqdm
import pickle
import subprocess
import sys
import yaml


def spawn_evals(cfg_path, cat):
    with open(cfg_path, 'r') as f:
        cfgs = yaml.safe_load(f)
    out_dir = os.path.join('out', cfgs['name'])

    print(f"out_dir: {out_dir}")

    #p_gen = subprocess.Popen(["sbatch", "./clusters/dist_gen.sh", cfg_path])
    #p_stdout, p_stderr = p_gen.communicate()
    res = subprocess.check_output(["sbatch", "./cluster/cli_gen.sh", cfg_path])
    res_str = res.decode('utf-8')
    job_id = res_str.rstrip('\n').split(' ')[-1]
    print(res)


    res_eval = subprocess.check_output(["sbatch", f"--depend=afterok:{job_id}", "./cluster/dist_cgan_eval.sh", out_dir, f"-c={cat}"])
    eval_str = res_eval.decode('utf-8')
    eval_id = eval_str.rstrip('\n').split(' ')[-1]
    print(res_eval)

    print("sbatch", f"--depend=afterok:{eval_id}", "./cluster/cgan_sum.sh", out_dir)
    res_sum = subprocess.check_output(["sbatch", f"--depend=afterok:{eval_id}", "./cluster/cgan_sum.sh", out_dir]) # Summarizer will summarize all categories by default
    print(res_sum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str)
    parser.add_argument("-c", "--category", type=str, help="chair, airplane, table, or cgan_all", default='cgan_all')
    args = parser.parse_args()

    spawn_evals(args.cfg_file, args.category)
