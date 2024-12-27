from glob import glob
import json
import numpy as np
import sys
import os
import os.path as osp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", type=str)
    parser.add_argument("-c", "--category", type=str, default='chair')

    args = parser.parse_args()

    exp_path = args.exp_path
    if args.category:
        CGAN_EVAL_CATS = [args.category]
    else:
        CGAN_EVAL_CATS = ['airplane', 'chair', 'table', ]

    res_dict = dict()

    for category in CGAN_EVAL_CATS:
        try:
            if osp.exists(f"{exp_path}/points/"):
                info_paths = glob(f'{exp_path}/points-eval_{category}_*.txt')
                # print("Using pre-sampled 2048 points for evaluation")
            else:
                info_paths = glob(f'{exp_path}/meshes-eval_{category}_*.txt')
            if len(info_paths) == 0:
                print(f"no record for {category}")

                res_dict[category] = {'uhd': np.nan, 'tmd': np.nan, 'comp': np.nan}
                continue

            eval_infos = list()
            for p in info_paths:
                with open(p, 'r') as f:
                    d = json.load(f)
                    eval_infos.append(d)

            # import ipdb; ipdb.set_trace()
            tmd = sum([i['tmd'] * i['num_items'] for i in eval_infos]) / sum([i['num_items'] for i in eval_infos])
            uhd = sum([i['uhd'] * i['num_items'] for i in eval_infos]) / sum([i['num_items'] for i in eval_infos])
            comp = sum([i['comp'] * i['num_items'] for i in eval_infos]) / sum([i['num_items'] for i in eval_infos])

            if 'total_items' in eval_infos[0]: # For compatibility
                total_items = sum(i['total_items'] for i in eval_infos)
                num_items = sum(i['num_items'] for i in eval_infos)
                if num_items != total_items:
                    print(f"WARNNING: only {num_items}/{total_items} are valid in {category}")

            print(f"{category}: tmd: {tmd}, uhd: {uhd}, comp: {comp}")
            res_dict[category] = {'uhd': uhd, 'tmd': tmd, 'comp': comp}
        except Exception as e:
            print(e)
            print(f"summarization failed on {category}")

    res_dict['avg'] = {'uhd': np.mean([i['uhd'] for i in res_dict.values()]),
            'comp': np.mean([i['comp'] for i in res_dict.values()]),
            'tmd': np.mean([i['tmd'] for i in res_dict.values()]),
            }
    print(f"avg: {res_dict['avg']}")
    with open(f"{exp_path}/summarized_eval.json", "w") as f:
        json.dump(res_dict, f)
