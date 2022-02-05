#!/bin/bash

python run.py --env Hopper-v6 \
              --seed 100 \
              --expert_db $1 \
              --num_trajs 10 \
              --n_models 2 \
              --lambda_b 0.1 \
              --samples_per_step 3200 \
              --n_dynamics_samples 3200 \
              --buffer_size 6400 \
              --bw_quantile 0.1 \
              --update_type exact \
              --cost_lr 0.0 \
              --id 1 \
              --n_iter 300 \
              --n_minmax 1 \
              --pg_iter 10 \
              --n_epochs 50 \
              --cg_iter 10 \
              --dynamics_model_hidden 512 512 \
              --num_cpu 1 \
              --grad_clip 4.0 \
              --norm_thresh_coeff 1 \
              --policy_min_log -1.0 \
              --policy_init_log -0.5 
