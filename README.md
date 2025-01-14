# ARPESNet
This repository contains the code for the paper "ACES-GNN: Can Graph Neural Network Learn to Explain Activity Cliffs".

## Dependencies
The code was developed and tested on Python 3.10.10 using CUDA 11.4 with
the following Python packages installed:
```
pytorch            1.12.1
torch-geometric    2.3.0
scikit-learn       1.2.2
captum             0.7.0
hyperopt           0.2.7
rdkit              2023.9.4
```

### Basic Usage
To run experiments, use the main script with appropriate parameters (see parsing.py under XACs/utils/ for details):

```bash
python main.py --dataset [dataset_name] \
               --config_dir [config_directory] \
               --data_dir [data_directory]
               --model_dir [model_directory] \
               --mode [mode] \
               --loss [loss_type] \
               --sim_threshold [similarity_threshold] \
               --dist_threshold [potency_distance_threshold]
               --conv_name [backbone]
```
### Example Commands

1. For CHEMBL214_Ki dataset:
```bash
python main.py --dataset 'CHEMBL214_Ki' \
               --config_dir './configs/nn_configs' \
               --data_dir 'Data'
               --model_dir './checkpoints/' \
               --mode 'train_test' \
               --loss 'MSE+direction' \
               --sim_threshold 0.9 \
               --dist_threshold 1.0 \
               --conv_name 'nn'
```