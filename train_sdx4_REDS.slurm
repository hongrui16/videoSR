#!/bin/bash

#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq' partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relevant QOS
#SBATCH --job-name=sdx4                   # Job name
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/zlog/zslurm/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/zlog/zslurm/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2                # One task per GPU
#SBATCH --gres=gpu:A100.80gb:2             # Request 2 A100 GPUs
#SBATCH --mem=80gb                         # Memory per node
#SBATCH --export=ALL
#SBATCH --time=4-24:00:00                  # Set to 4 days and 24 hours
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task



SRC_CFG="/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/config/config_SDX4.py"



CURRENT_TIME=$(date +%F-%H-%M-%S)

# （可选）把最终配置备份到日志目录
LOG_DIR="/home/rhong5/research_pro/hand_modeling_pro/fsv/vsr/zlog/SD-X4/REDS/${CURRENT_TIME}_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOG_DIR"
cp "$0" ${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.slurm

# 为本作业创建影子包
JOBCFG="${LOG_DIR}/.jobcfg_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOBCFG/config"

# 复制为 __init__.py（也可 ln -s 软链）
cp "$SRC_CFG" "$JOBCFG/config/__init__.py"

# 让影子包优先被导入
export PYTHONPATH="$JOBCFG:$PYTHONPATH"




# cp "$JOBCFG/config/__init__.py" "$LOG_DIR/config_REDS_${SLURM_JOB_ID:-$$}.py"
export LOG_DIR




WEIGHT_DIR="/scratch/rhong5/weights/temp_training_weights/SD-X4/REDS/${CURRENT_TIME}_${SLURM_JOB_ID:-$$}"
export WEIGHT_DIR



module load gnu10
module load python



source /home/rhong5/py310Torch/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/fsv/vsr


PORT=$(( (RANDOM << 8 | RANDOM) & 0xFFFF ))
echo "Random port: $PORT"



# Launch distributed training with accelerate
accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 --main_process_ip=localhost --main_process_port=$PORT \
    train_sdx4.py 2> >(grep -v "absl::InitializeLog" | grep -v "gl_context_egl.cc" | grep -v "gl_context.cc" >&2)





# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00