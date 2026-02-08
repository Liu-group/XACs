# ACES-GNN
This repository contains the code for the paper "ACES-GNN: Can Graph Neural Network Learn to Explain Activity Cliffs".

## Dependencies
The code was developed and tested on Python 3.10.10 using CUDA 11.4 with the following Python packages installed:
```
pytorch            1.12.1
torch-geometric    2.3.0
scikit-learn       1.2.2
captum             0.7.0
hyperopt           0.2.7
rdkit              2023.9.4
```

## Dataset Preparation
Place your datasets in the `Data/` directory. The expected structure is:
```
Data/
├── CHEMBL214_Ki/
├── [other_datasets]/
└── ...
```

## Configuration
Configuration files should be placed in the `configs/nn_configs/` directory. These files contain hyperparameters and model settings for different experiments.

## Basic Usage
To run experiments, use the main script with appropriate parameters (see parsing.py under XACs/utils/ for details):

```bash
python main.py --dataset [dataset_name] \
               --config_dir [config_directory] \
               --data_dir [data_directory] \
               --model_dir [model_directory] \
               --loss [loss_type] \
               --sim_threshold [similarity_threshold] \
               --dist_threshold [potency_distance_threshold] \
               --conv_name [backbone]
```

### Example Commands

Train and test MPNN with supervision loss for CHEMBL214_Ki dataset:
```bash
python main.py --dataset 'CHEMBL214_Ki' \
               --config_dir './configs/nn_configs' \
               --data_dir 'Data' \
               --model_dir './checkpoints/' \
               --loss 'MSE+direction' \
               --sim_threshold 0.9 \
               --dist_threshold 1.0 \
               --conv_name 'nn'
```

## Reproducing Paper Experiments

### Batch Experiments with SLURM
For reproducing the full experiments from the paper, use the `run_task.py` script which automates running experiments across multiple datasets and model configurations:

```bash
python run_task.py
```

**Important**: Before running, update the conda environment name in `run_task.py`:
```python
sub_sh.write('conda activate YOUR_ENVIRONMENT_NAME\n')  # Change to your environment name
```

This script will:
- Run experiments with different GNN backbones (GAT, GINE, MPNN)
- Test both loss functions ('MSE' and 'MSE+direction')
- Process all datasets defined in `XACs.utils.const.DATASETS`
- Submit jobs to SLURM scheduler (modify SLURM parameters as needed for your cluster)
- Organize results in `./results/results_cv/` and `./results/results_cv_x/` directories

### Manual Batch Execution
If you don't have SLURM, you can modify `run_task.py` to run experiments sequentially by replacing the `subprocess.call(['sbatch', ...])` lines with direct execution.

### Results Analysis
The `notebooks/` folder contains Jupyter notebooks for analyzing experimental results:
- Data visualization and statistical analysis
- Performance comparison across different models
- Activity cliff explanation analysis
- Reproduction of paper figures and tables

To use the notebooks:
```bash
jupyter notebook notebooks/
```

## Project Structure
```
XACs/
├── main.py                      # Main entrypoint (cross-validation train/eval)
├── run_task.py                  # Batch runner (SLURM submission helper)
├── hypertune.py                 # Hyperparameter tuning (hyperopt + grid search)
├── requirement.txt              # Python dependencies (see Dependencies section)
├── Data/                        # Dataset directory (place datasets here)
├── configs/                     # Per-backbone configuration files
│   ├── nn_configs/
│   ├── gat_configs/
│   └── gine_configs/
├── results/                     # Outputs/checkpoints (e.g., results_cv/, results_cv_x/)
├── notebooks/                   # Jupyter notebooks for analysis/plots
├── XACs/                        # Core library code (models, training, evaluation, utilities)
│   ├── attribution/             # Attribution methods (e.g., GradCAM)
│   ├── models/                  # GNN backbones/architectures
│   └── utils/                   # CLI parsing, constants, helpers
└── README.md
```

## Output
- Model checkpoints are saved in the `--model_dir` directory
- Training logs and results are typically saved alongside the checkpoints
- Batch experiment results are organized in `./results/` with subdirectories for different loss functions and datasets