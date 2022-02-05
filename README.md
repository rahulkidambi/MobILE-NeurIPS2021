# MobILE-NeurIPS2021
This is the code repository for MobILE: Model-Based Imitation Learning from Observation Alone (NeurIPS 2021)

![figure](https://github.com/rahulkidambi/MobILE-NeurIPS2021/blob/main/figures/plot.png)

Link to pdf: https://arxiv.org/abs/2102.10769

## Note on Environment Setup
Install openai gym by running `cd gym && pip install -e .`

Install mjrl by running `cd mjrl && pip install -e .`

Install mbil by running `cd mbil && pip install -e .`

Additional dependencies are listed in `requirements.txt`. The experiments are run using MuJoCo physics, which requires a license to install. Please follow the instructions on [MuJoCo Website](http://www.mujoco.org)

## Overview
`mjrl` is our policy gradient library that we modified to inferface with our cost functions when doing model-based policy gradient. This modification can be seen in `mjrl/mjrl/algos/batch_reinforce.py`. We use TRPO/NPG by default, but in principle one could replace TRPO/NPG with other algorithms/repositories.

`mbil` (model based imitation learning) is where we have most of the components. 
1. cost: directory where cost functions go
2. dynamics_model: directory with the dynamics model ensembles
3. dataset: utility/object for creating datasets
4. env: contains the model-based wrappers. Add to `wrappers.py`
5. utils: modify/add to `arguments.py` for argparse args

## Current Environments Supported
This repository supports 2 modified MuJoCo environments. They are

- Hopper (Hopper-v6, deterministic)
- Walker2d (Walker2d-v4, deterministic)

If you would like to add an environment, register the environment in `/milo/milo/gym_env/__init__.py` according to [OpenAI Gym](http://gym.openai.com/docs/#environments) instructions.

## Dataset Format
In the `data` directory, place the expert dataset in the `data/expert_data` directory. Note to modify the dataset format please see `mbil/utils/util.py`. Generally we only expect a tuple of state and next state from the expert demonstrations. 

## Running an Experiment
We provide an example run script for Hopper, `example/run_hopper.sh`, that can be modified to be used with any other registered environment. To view all the possible arguments you can run please see the argparse in `mbil/utils/arguments.py`. Note you would need to approprietly set the dataset path.

## Bibliography
To cite this work, please use the following citation. Note that this repository builds upon MJRL so please also cite any references noted in the README [here](https://github.com/aravindr93/mjrl).
```
@article{kidambi2021mobile,
      title={MobILE: Model-Based Imitation Learning from Observations Alone},
      author={Rahul Kidambi and Jonathan D. Chang and Wen Sun},
      year={2021},
      eprint={2102.10769},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

