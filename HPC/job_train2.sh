#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100 
### -- set the job Name -- 
#BSUB -J Training_data
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 24GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 03:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o TrainREG_%J.out 
#BSUB -e TrainREG_%J.err
### -- Load modules needed by crea_dataset.py

# The CUDA device reserved for you by the batch system
#CUDADEV=`cat $PBS_GPUFILE | rev | cut -d"-" -f1 | rev | tr -cd [:digit:]`


nvidia-smi

/appl/cuda/10.0/samples/bin/x86_64/linux/release/deviceQuery

### -- Run program
#python train8.py
#python train9.py
#python train10.py
#python train11.py
#python train14.py
python train15.py 
