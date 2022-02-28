#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -c 16
#SBATCH --job-name=HandEggNEAT
#SBATCH --output=output.txt

conda activate openai-gym
cd /home/w090rjs/Personal/Data-Analysis/OpenAI-Gym/Experiments/NEAT
python HandManiuplateEgg-NEAT.py