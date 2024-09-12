import os
import pandas as pd
CPUs = os.cpu_count()

def convert_csv_to_smi(csv_file, smi_file = None):
    """
    Convert csv file to smi file
    """
    data = pd.read_csv(csv_file)
    smiles_data = data['smiles']
    smi_file = csv_file.replace('.csv', '.smi') if smi_file is None else smi_file
    smiles_with_id = pd.DataFrame({
            'smiles': smiles_data,
            'id': range(1, len(smiles_data) + 1)  # Adding a simple sequential identifier
            })
    smiles_with_id.to_csv(smi_file, sep=' ', index=False, header=False)
    print(f"SMILES data saved to {smi_file}")
    return smi_file

def calc_MMPs(smifile, fragfile = None, mmp_path = None):
    """
    Generate MMP Indexing and Matching using mmpdb
    """
    # TODO switch system calls to just importing the python code and using it directly
    print("Generating MMP Fragments")
    fragfile = smifile.replace('.smi', '.frag') if fragfile is None else fragfile
    # add mmp to the name of the smifile
    mmpdb_out = smifile.split('.')[0] + '_mmp.csv' if mmp_path is None else mmp_path
    os.system(f'mmpdb fragment {smifile} --num-jobs {CPUs} --num-cuts 1 -o {fragfile}')
    os.system(f"mmpdb index {fragfile} -s\
                                     --max-variable-ratio 0.33 \
                                     --max-heavies-transf 8 \
                                     -o {mmpdb_out} \
                                     --max-variable-heavies 13 \
                                     --out 'csv'")
    print(f"MMPs saved to {mmpdb_out}")
    return mmpdb_out

if __name__ == "__main__":
    csv_file = '/home/xuchen/ACs/Data/BACE/BACE.csv'
    smi_file = convert_csv_to_smi(csv_file)
    mmpdb_out = calc_MMPs(smi_file)
    mmpdb_out_df = pd.read_csv(mmpdb_out, sep='\t', names=['smi_1', 'smi_2', 'idx1', 'idx2', 'transformation', 'core'])
    if len(mmpdb_out_df) == 0:
        print(f'For target {target_id}, there is no MMP generated after mmpdb.')
        failed_targets.append(target_id)
    print(f"Number of MMPs generated: {len(mmpdb_out_df)}")
    data = pd.read_csv(csv_file)
    data = data[['smiles', 'y']]
    merged1 = mmpdb_out_df.merge(data.rename(columns={'smiles': 'smi_1', 'y': 'y1'}), on='smi_1')
    merged2 = merged1.merge(data.rename(columns={'smiles': 'smi_2', 'y': 'y2'}), on='smi_2')
    merged2['y_diff'] = merged2['y1'] - merged2['y2']
    # remove y_diff = 0
    merged2 = merged2[merged2['y_diff'] != 0]
    merged2.to_csv(mmpdb_out.replace('.csv', '_with_y.csv'), index=False)
