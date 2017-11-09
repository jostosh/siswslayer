#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=LWS
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output LWS-%j.log
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow
source envs/lws/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/s2098407/packages/hypylib

srun python siswslayer/train_keras.py --data_path /data/s2098407/adience $*