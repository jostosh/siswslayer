#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=CIFAR10
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output CIFAR10-%j.log
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow
source envs/lws/bin/activate

srun python siswslayer/train_keras.py --dataset cifar10 --kerosene_path /data/s2098407/kerosene $*