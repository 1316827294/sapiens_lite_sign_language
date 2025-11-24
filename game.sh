#!/usr/bin/bash -l

### SBATCH parameters for GPU job
#SBATCH --time=0-50:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=16G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=4          ## 2 CPUs
#SBATCH --gpus=T4:1                ## 1 T4 GPU
#SBATCH --job-name=game_video2pose        ## Job name
#SBATCH --output=game_video2pose_%j.out   ## Output file (%j = job ID)
#SBATCH --error=game_video2pose_%j.err    ## Error file (%j = job ID)

# Load environment modules
# Load environment modules
module load anaconda3
# Activate conda environment
source activate sapiens_lite


# Run the Python program
python runvideo.py