#!/bin/bash

#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq' partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relevant QOS
#SBATCH --job-name=vsr                   # Job name
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/zlog/zslurm/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/zlog/zslurm/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3                # One task per GPU
#SBATCH --gres=gpu:A100.80gb:3             # Request 2 A100 GPUs
#SBATCH --mem=80gb                         # Memory per node
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00                  # Set to 4 days and 24 hours
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task

module load gnu10
module load python


source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/fsv/vsr


PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"

DATASET_NAME="Vimeo-90K"
unet_weights_precision="fp32"


# Launch distributed training with accelerate
accelerate launch --num_processes=3 --num_machines=1 --machine_rank=0 --main_process_ip=localhost --main_process_port=$PORT \
    train_sdx4_fri_dstcm.py --dataset_name=$DATASET_NAME --unet_weights_precision=$unet_weights_precision 2> >(grep -v "absl::InitializeLog" | grep -v "gl_context_egl.cc" | grep -v "gl_context.cc" >&2)


# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00