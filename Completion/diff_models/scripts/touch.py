from glob import glob
import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    d = glob(f"{sys.argv[1]}/**", recursive=True)
    for p in tqdm(d):
        os.system(f"touch {p}")
