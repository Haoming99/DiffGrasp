from glob import glob
import json
import sys

if __name__ == '__main__':
    exp_path = sys.argv[1]

    res_dict = dict()

    info_paths = glob(f'{exp_path}/eval_*.json')

    eval_infos = list()
    for p in info_paths:
        with open(p, 'r') as f:
            d = json.load(f)
            eval_infos.append(d)

    # breakpoint()
    tmd = sum([i['tmd'] * i['num_items'] for i in eval_infos]) / \
        sum([i['num_items'] for i in eval_infos])
    uhd = sum([i['uhd'] * i['num_items'] for i in eval_infos]) / \
        sum([i['num_items'] for i in eval_infos])
    min_chamfer_l1 = sum([i['min_chamfer_l1'] * i['num_items']
                         for i in eval_infos]) / sum([i['num_items'] for i in eval_infos])
    mean_chamfer_l1 = sum([i['mean_chamfer_l1'] * i['num_items']
                          for i in eval_infos]) / sum([i['num_items'] for i in eval_infos])

    total_items = sum(i['total_items'] for i in eval_infos)
    num_items = sum(i['num_items'] for i in eval_infos)
    if num_items != total_items:
        print(f"WARNNING: only {num_items}/{total_items} are valid")

    print(f"tmd: {tmd*1e4}, uhd: {uhd}, mean_chamfer_l1: {mean_chamfer_l1*1e4}, min_chamfer_l1: {min_chamfer_l1 * 1e4}")
    res_dict = {'uhd': uhd, 'tmd': tmd,
                'mean_chamfer_l1': mean_chamfer_l1, 'min_chamfer_l1': min_chamfer_l1}

    with open(f"{exp_path}/summarized_eval.json", "w") as f:
        json.dump(res_dict, f)
