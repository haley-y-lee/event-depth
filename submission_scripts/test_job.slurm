#!/bin/bash
#SBATCH -J test_file                         # Job name
#SBATCH -o test_file_%j.out                  # output file (%j expands to jobID)
#SBATCH -e test_file_%j.err                  # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=abc256@cornell.edu        # Email address to send results to.
#SBATCH -n 4                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=40gb                           # server memory requested (per node)
#SBATCH -t 96:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=monakhova               # Request partition
#SBATCH --gres=gpu:1                        # Type/number of GPUs needed

source /home/abc256/miniconda3/etc/profile.d/conda.sh

conda activate mono_train

export PREPROCESSED_DATASETS_FOLDER="/home/abc256/mono/DENSE"

cd mono/mono_event/rpg_e2depth/e2depth/

python train.py -c configs/test_config.json



