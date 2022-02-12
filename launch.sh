export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
source ~/.bashrc
conda activate habitat
srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/transformer.yaml --run-type train