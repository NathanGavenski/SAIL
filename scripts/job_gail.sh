#! /bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=BC_%j

echo "ACTIVATE BASH"
source /users/${USER}/.bashrc
source activate /scratch/users/${USER}/conda/adversarial

echo "\nMODULES"
module load mesa-glu/9.0.2-gcc-9.4.0 
module load cuda/11.4.2-gcc-9.4.0
module load cudnn/8.2.4.15-11.4-gcc-9.4.0
module list

echo "\nACTIVATING ENV"

echo "\nRUNNING EXPERIMENT"
cd /users/k21158663/BCIO-Torch-Implementation
python gail_training.py --env $1 --data $2 --batch $3