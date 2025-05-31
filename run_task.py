import os
import subprocess 
from XACs.utils.const import DATASETS

def write_sub_file(results_dir, job_name, configurations): 
    #write sub.sh file
    sub_sh = open(f'{results_dir}/sub_{job_name}.sh','w')
    sub_sh.write('#!/bin/bash\n')
    sub_sh.write('#SBATCH --time=1-00:00:00\n')
    sub_sh.write('#SBATCH --nodes=1\n') 
    sub_sh.write('#SBATCH --mem=2G\n')
    sub_sh.write('#SBATCH --ntasks=1\n')
    sub_sh.write('#SBATCH --cpus-per-task=1\n')
    sub_sh.write('#SBATCH --gres=gpu:1\n')
    sub_sh.write('#SBATCH --partition=day-long\n')
    sub_sh.write('#SBATCH --job-name=%s\n' % job_name)
    sub_sh.write('#SBATCH --output=%s/R-%s.out\n' % (results_dir, job_name))
    sub_sh.write('conda activate YOUR_ENVIRONMENT_NAME\n')    
    for config in configurations:
        dataset, backbone, loss = config 
        config_dir = f'./configs/{backbone}_configs'
        os.makedirs(config_dir, exist_ok=True)
        model_dir = f'./results/results_{task}/{backbone}'
        os.makedirs(model_dir, exist_ok=True)
        sub_sh.write('python main.py --dataset %s --loss %s --config_dir %s --model_dir %s --conv_name %s\n' % (dataset, loss, config_dir, model_dir, backbone))
    sub_sh.close()
    return job_name

if __name__ == '__main__':
    configurations = []
    max_configs_per_file = 1 # Number of configurations to bundle into one .sh file
    job_idx = 1
    loss = ['MSE', 'MSE+direction']
    for l in loss:
        for backbone in ['gat', 'gine', 'nn']: # gine, gat, nn, dmpnn, pna
            task = 'cv' if l == 'MSE' else 'cv_x'
            results_dir = f'./results/results_{task}/{backbone}'
            os.makedirs(results_dir, exist_ok=True)
            for dataset in DATASETS:  # You can adjust datasets or use DATASETS/MOLDATASETS
                configurations.append((dataset, backbone, l))
                # If we've reached the max configs per file, write to a new sub file
                if len(configurations) >= max_configs_per_file:
                    job_name = f'{backbone}_batch_job_{job_idx}'
                    write_sub_file(results_dir, job_name, configurations)
                    subprocess.call(['sbatch', f'{results_dir}/sub_{job_name}.sh'])
                    print(f"Submitted {job_name} with {len(configurations)} configurations")
                    job_idx += 1
                    configurations = []

    # Submit remaining configurations if they exist
    if configurations:
        job_name = f'{backbone}_batch_job_{job_idx}'
        write_sub_file(results_dir, job_name, configurations)
        subprocess.call(['sbatch', f'{results_dir}/sub_{job_name}.sh'])
        print(f"Submitted the rest {job_name} with {len(configurations)} configurations")
