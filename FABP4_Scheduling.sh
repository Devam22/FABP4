#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00-08:00
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-jtrant
#SBATCH --output=fabp4_part2.log

# Load required modules
module load StdEnv/2023 gcc/12.3 python/3.11.5 rdkit/2023.09.3 cuda/12.2 cudnn/8.9.5

# Create and activate virtual environment
virtualenv --no-download --python=$(which python3.11) $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip and install required packages
pip install --no-index --upgrade pip
pip install --no-index scikit-learn numpy pandas keras xgboost

# Install RAPIDS suite (adjust versions as needed)
pip install --no-index cudf-cu12 cuml-cu12 cupy-cuda12x dask distributed

# Install additional required packages
pip install --no-index joblib

# Run your Python script
python /project/6007964/cache5_library/Pipeline_dv.py