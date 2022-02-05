# MobILE-NeurIPS2021
This is the code repository for MobILE: Model-Based Imitation Learning from Observation Alone (NeurIPS 2021)

## Current Environments Supported
- Hopper (Hopper-v6, deterministic)
- Walker2d (Walker2d-v4, deterministic)
- HalfCheetah (HalfCheetah-v4, deterministic)

## Current Online PG algorithms Supported
- TRPO

## Note on Environment Setup
Install openai gym by running `cd gym && pip install -e .`

Install mjrl by running `cd mjrl && pip install -e .`

Install mbil by running `cd mbil && pip install -e .`

## General Organization
`mjrl` is our policy gradient library. Using TRPO/NPG by default, PPO coming

`mbil` (model based imitation learning) is where we have most of the components. 
1. cost: directory where cost functions go
2. dynamics_model: directory with the dynamics model ensembles
3. dataset: utility/object for creating datasets
4. env: contains the model-based wrappers. Add to `wrappers.py`
5. utils: modify/add to `arguments.py` for argparse args

## Input Dataset Format

