#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=ADIENCE
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output ADIENCE-%j.log
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow
source envs/lws/bin/activate

srun python siswslayer/train_keras.py --dataset adience --kerosene_path /data/s2098407/kerosene $*