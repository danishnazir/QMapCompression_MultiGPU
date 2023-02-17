import pandas as pd
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='path for dataset', required=True)
parser.add_argument('--name', help='name for dataset', required=True)
args = parser.parse_args()

exts = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
data_dir = '../data'

sub_dirs = os.listdir(args.dir)



os.makedirs(data_dir, exist_ok=True)

dataset_path = []

for _dir in sub_dirs:

    pth = os.path.join(args.dir, _dir)
    for ext in exts:
        dataset_path += glob.glob(os.path.join(pth, f'*.{ext}'))

pd.DataFrame(dataset_path, columns=['path']).to_csv(f'{data_dir}/{args.name}.csv', index=False)
print(len(dataset_path))
