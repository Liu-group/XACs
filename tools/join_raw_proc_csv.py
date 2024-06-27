import sys
import os
# add ../ to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from const import DATASETS
import pandas as pd


if __name__ == '__main__':
    for name in DATASETS:
        df_proc = pd.read_csv(f'../Benchmark_Data/{name}/{name}.csv')
        df_proc.drop(columns=['split'], inplace=True)
        df_raw = pd.read_csv(f'../Benchmark_Data_Raw/{name}.csv')
        df_raw.drop(columns=['Unnamed: 0', 'exp_mean [nM]', 'class'], inplace=True)
        df_new = df_proc.merge(df_raw, on='smiles', how='left')
        df_new.to_csv(f'../Benchmark_Data/{name}/{name}.csv', index=False)
