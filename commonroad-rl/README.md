# CommonRoad-RL

This repository contains a software package to solve motion planning problems on [CommonRoad](https://commonroad.in.tum.de) 
using Reinforcement Learning algorithms, currently from [OpenAI Stable Baselines](https://stable-baselines.readthedocs.io/en/master/).

## Folder structure
```
commonroad-rl                                           
├─ commonroad_rl                                        
│  ├─ gym_commonroad                    # Gym environment for CommonRoad scenarios
|     ├─ feature_extraction             # Functions to extract goal-related and surrounding-related observations
|     ├─ utils                          # Functions to calculate other observations
│     ├─ configs.yaml                   # Default config file for observation space, reward definition, and termination conditions, 
                                          as well as for observation space optimization and reward coefficient optimization
│     ├─ commonroad_env.py              # CommonRoadEnv(gym.Env) class
│     ├─ constants.py                   # Script to define path, vehicle, and draw parameters
│     └─ vehicle.py                     # All supported vehicle models                  
│  ├─ hyperparams                       # Config files for default hyperparameters for various RL algorithms                                       
│  ├─ tests                             # Test system of commmonroad-rl.
│  ├─ tools                             # Tools to validate, visualize and analyze CommonRoad .xml files, as well as preprocess and convert to .pickle files.                 
│  ├─ utils_highd                       # Tools to convert raw highD dataset files to CommonRoad .xml files                      
│  ├─ utils_ind                         # Tools to convert raw inD dataset files to CommonRoad .xml files                        
│  ├─ utils_run                         # Utility functions to run training, tuning and evaluating files                                      
│  ├─ README.md                                                      
│  ├─ play_stable_baselines.py          # Script to evaluate a trained RL model on specific scenarios and visualize the scenario                
│  ├─ run_stable_baselines.py           # Script to train RL model or optimize hyperparameters or environment configurations           
│  ├─ solve_stable_baselines.py         # Script to genearte CommonRoad solution files from trained RL models.
│  └─ plot_results_stable_baselines.py  # Plot learning curves with provided training log files.                
├─ scripts                              # Bash scripts to install all dependencies, train and evaluate RL models, as well as generate CommonRoad solution files from trained RL models.                                                                                        
                 
├─ README.md                                            
├─ commonroad_style_guide.rst           # Coding style guide for this project                
├─ environment.yml                                      
└─ setup.py                                             
```
## Installation

### Prerequisites 
This project should be run with conda. Make sure it is installed before proceeding with the installation.

To create an environment for this project including all requirements, run
```
conda env create -n cr36 -f environment.yml
```

### Install without sudo rights

Simply run
```
bash scripts/install.sh -e cr36 --no-root
```
`cr36` to be replaced by the name of your conda environment if needed.

This will build all softwares in your home folder. You can press `ctrl+c` to skip when asked for sudo password.
Note that all necessary libraries need to be installed with sudo rights beforehands. 
Please ask your admin to install them for you if they are missing.


### Install with sudo rights
Simply run
```
bash scripts/install.sh -e cr36
```
`cr36` to be replaced by the name of your conda environment if needed.


### Test if installation succeeds

Further details of our test system refer to [here](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl/-/tree/development/commonroad_rl/tests).

```
source activate cr36
pytest commonroad_rl/tests --scope unit module -m "not slow"
```

## Usage
Please check [tutorials](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl/-/tree/development/tutorials) for further details.

## Development

Please use `development` branch or open a new `feature_xxx` branch to make contribution.

## References and Suggested Guides

1. Krasowski, Hanna; Wang, Xiao; Althoff, Matthias: Safe Reinforcement Learning for Autonomous Lane Changing Using Set-Based Prediction. 2020 IEEE International Conference on Intelligent Transportation Systems (ITSC), 2020. *[[Fulltext (mediaTUM)]](https://mediatum.ub.tum.de/doc/1548735/256213.pdf)*   
2. [OpenAI Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)
3. [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
4. [OpenAI Gym](https://gym.openai.com/docs/)
5. [OpenAI Safety Gym](https://openai.com/blog/safety-gym/)
