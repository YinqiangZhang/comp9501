# COMP9501 Project: Reinforcement Learning for Autonomous Driving in Urban Scenarios

**Author:** Yinqiang ZHANG, Wentao CHEN

---

This project provides codes with comments to set up an Reinforcement Learning (RL) environemnt for the course project COMP9501. A hierarchical discrete action space is used for autonomous driving in urban scenarios. This is a simple test of our machine learning course.  

# Getting Started

The software is written in Python 3.6 and tested on Linux (Ubuntu 18.04). The usage of the Anaconda Python distribution is strongly recommended. 

## Prerequisites

1. **commonroad-rl**

The commonroad-rl package provides various methods to construct basic environment. Please follow the instruction in commonroad-rl package to intall it. For installation, please refer to [instructions](https://commonroad.in.tum.de/commonroad-rl). 

2. **stable-baselines**

The code for RL algorithms is adapted from standard [Stable Baselines](https://stable-baselines.readthedocs.io/). Based on the [previous adapted version](https://github.com/NTUT-SELab/stable-baselines/tree/ActionMask), slight modifications are added. After installing commonroad-rl, the default installation of stable_baselines should be removed. Then the path of stable-baselines folder should be included in the python searching path. 

## Installing

It is assumed that you have installed Anaconda and that your Anaconda environment is called **auto-rl**.
 
1. Open your terminal in the root path of this project.

2. Following instructions in the prerequisites: 
- Switch to *./commonroad-rl*, create Anaconda environment and install the entire commonroad-rl package as follows:

```
conda env create -n auto_rl -f environment.yml
```

 This installation includes the basic configurations of **auto-rl** environment and following packages: commonroad-io, commonroad drivability checker, commonroad curvilinear coordinate system, and route planner. If possible, please use the [shell script](./commonroad-rl/scripts/install.sh) to install the entire package directly with following command for root user: 

```
bash scripts/install.sh -e auto_rl
```

If no root, an alternative method is here:

```
bash scripts/install.sh -e demo_env --no-root
```

- Download InD dataset and generate InD scenarios according to [the guidance](./commonroad-rl/commonroad_rl/utils_ind/README.md) in **commonroad-rl/commonroad_rl/utils_ind**. For the details of InD dataset, please refer to [here](https://www.ind-dataset.com/).

- Add folders of stable-baselines in the searching path of environment **demo_env**. (please remove the standard stable-baselines installed by commonroad-rl package in advance)
   
### Test your installation 

1. Activate your environment with ```conda activate demo_env```

2. Go to the the folder */your/root/path/commonroad-rl/commonroad_rl/gym_commonroad* and adapt the path parameters in [constants.py](./commonroad-rl/commonroad_rl/gym_commonroad/constants.py) and open the [project_configs.yml](./project_configs.yml) in root path of project to set hyperparameters (*especially ROOT_PATH*) for running the project. 

3. run the following three demos:
- run [environement_testing_demo.py](./environement_testing_demo.py) to test the training environment
- run [policy_running_demo.py](./policy_running_demo.py) to run the learned policy in *training_results/demo_policy*. 
- run [policy_training_demo.py](./policy_training_demo.py) to training a policy. 
