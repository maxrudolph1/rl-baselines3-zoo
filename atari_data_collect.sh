#!/bin/bash

# Parameters
#SBATCH --account=amyzhang
#SBATCH --cpus-per-task=64
#SBATCH --error=slurm_scripts/job_%j/err.err
#SBATCH --output=slurm_scripts/job_%j/out.out
#SBATCH --gpus-per-node=1
#SBATCH --job-name=atari
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=allnodes
#SBATCH --time=1200

source /u/mrudolph/miniconda3/etc/profile.d/conda.sh
conda activate rlzoo

# games
STEP=100000

GAME=BreakoutNoFrameskip-v4
# GAME=AsteroidsNoFrameskip-v4
# GAME=BeamRiderNoFrameskip-v4
# GAME=EnduroNoFrameskip-v4
# GAME=PongNoFrameskip-v4
# GAME=QbertNoFrameskip-v4
# GAME=SeaquestNoFrameskip-v4
# GAME=SpaceInvadersNoFrameskip-v4
# GAME=RoadRunnerNoFrameskip-v4

python enjoy.py --algo a2c --env $GAME --folder rl-trained-agents/ -n $STEP
python enjoy.py --algo dqn --env $GAME --folder rl-trained-agents/ -n $STEP
python enjoy.py --algo ppo --env $GAME --folder rl-trained-agents/ -n $STEP
python enjoy.py --algo qrdqn --env $GAME --folder rl-trained-agents/ -n $STEP