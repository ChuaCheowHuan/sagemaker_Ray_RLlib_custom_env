import json
import os

import gym
import ray
from ray.tune import run_experiments
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from mod_op_env import ArrivalSim

from sagemaker_rl.ray_launcher import SageMakerRayLauncher
        
"""
def create_environment(env_config):
    import gym
#     from gym.spaces import Space
    from gym.envs.registration import register

    # This import must happen inside the method so that worker processes import this code
    register(
        id='ArrivalSim-v0',
        entry_point='env:ArrivalSim',
        kwargs= {'price' : 40}
    )
    return gym.make('ArrivalSim-v0')
"""
def create_environment(env_config):
    price = 30.0
    # This import must happen inside the method so that worker processes import this code
    from mod_op_env import ArrivalSim
    return ArrivalSim(price)


class MyLauncher(SageMakerRayLauncher):
    def __init__(self):        
        super(MyLauncher, self).__init__()
        self.num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
        self.hosts_info = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))["hosts"]
        self.num_total_gpus = self.num_gpus * len(self.hosts_info)
        
    def register_env_creator(self):
        register_env("ArrivalSim-v0", create_environment)

    def get_experiment_config(self):
        return {
          "training": {
            "env": "ArrivalSim-v0",
            "run": "PPO",
            "stop": {
              "training_iteration": 3,
            },
              
            "local_dir": "/opt/ml/model/",
            "checkpoint_freq" : 3,
              
            "config": {                                
              #"num_workers": max(self.num_total_gpus-1, 1),
              "num_workers": max(self.num_cpus-1, 1),
              #"use_gpu_for_workers": False,
              "train_batch_size": 128, #5,
              "sample_batch_size": 32, #1,
              "gpu_fraction": 0.3,
              "optimizer": {
                "grads_per_step": 10
              },
            },
            #"trial_resources": {"cpu": 1, "gpu": 0, "extra_gpu": max(self.num_total_gpus-1, 1), "extra_cpu": 0},
            #"trial_resources": {"cpu": 1, "gpu": 0, "extra_gpu": max(self.num_total_gpus-1, 0),
            #                    "extra_cpu": max(self.num_cpus-1, 1)},
            "trial_resources": {"cpu": 1,
                                "extra_cpu": max(self.num_cpus-1, 1)},              
          }
        }

if __name__ == "__main__":
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
    os.environ["RAY_USE_XRAY"] = "1"
    print(ppo.DEFAULT_CONFIG)
    MyLauncher().train_main()
