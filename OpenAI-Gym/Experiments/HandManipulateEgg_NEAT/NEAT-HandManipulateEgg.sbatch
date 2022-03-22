#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -c 16
#SBATCH --job-name=HandEggNEAT
#SBATCH --output=output.txt


__conda_setup="$('/home/w090rjs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/w090rjs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/w090rjs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/w090rjs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup


conda activate openai-gym
cd /home/w090rjs/Personal/Data-Analysis/OpenAI-Gym/Experiments/NEAT
python HandManiuplateEgg-NEAT.py