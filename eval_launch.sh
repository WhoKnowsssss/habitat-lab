source ~/.bashrc
conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/habitat_new
srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/transformer.yaml --run-type eval SENSORS "('THIRD_RGB_SENSOR', 'HEAD_DEPTH_SENSOR', 'HEAD_RGB_SENSOR')" TASK_CONFIG.SIMULATOR.DEBUG_RENDER True TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS "('HEAD_DEPTH_SENSOR', 'THIRD_RGB_SENSOR', 'HEAD_RGB_SENSOR')" TASK_CONFIG.SIMULATOR.DEBUG_RENDER True
