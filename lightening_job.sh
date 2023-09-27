#!/bin/sh

# Example GPU job submission script

# This is a comment.
# Lines beginning with the # symbol are comments and are not interpreted by 
# the Job Scheduler.

# Lines beginning with #SBATCH are special commands to configure the job.
		
### Job Configuration Starts Here #############################################

# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env 
			
# The default is one task per node
#SBATCH --ntasks=1
#SBATCH --nodes=1

# Request 1 GPU
# Each gpu node has two logical GPUs, so up to 2 can be requested per node
# To request 2 GPUs use --gres=gpu:pascal:2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1 

#request 10 minutes of runtime - the job will be killed if it exceeds this
#SBATCH --time=2000:00

# Change email@example.com to your real email address
#SBATCH --mail-user=siddhi.bajracharya@coyotes.usd.edu
#SBATCH --mail-type=ALL

#SBATCH -e stderr.txt
#SBATCH -o stdout.txt
#SBATCH --open-mode=append




### Commands to run your program start here ####################################

#module add python/3.9.12-gcc-8.5.0 cuda/11.8 neovim/0.8.0-gcc-8.5.0 apptainer
#jupyter nbconvert --execute --to notebook multimodal_29-lightening.ipynb
#python multimodal-lightening-cough.py

~/anaconda3/envs/jupyter_clone/bin/python  emotion_utils/Audio-spectrogram-transformer-Classification-bert.py
~/anaconda3/envs/jupyter_clone/bin/python  emotion_utils/Audio-spectrogram-transformer-Classification.py
~/anaconda3/envs/jupyter_clone/bin/python  emotion_utils/Classification-with-language-model.py
