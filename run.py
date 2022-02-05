import os
import gym
import time
import pdb
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from mjrl.algos.npg_cg import NPG
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.behavior_cloning import BC
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths, sample_data_batch

import mbil.gym_env
from mbil.utils import *
from mbil.gym_env import model_based_env
from mbil.dataset import OnlineDataset
from mbil.cost import RBFLinearCost
from mbil.dynamics_model import DynamicsEnsemble

def main():
    args = get_args()
    dirs, ids, ensemble_checkpoint, logger, writer, device = setup(args, ask_prompt=True)

    # ======== Dataset Setup ==========
    expert_db_path = os.path.join(args.data_path, 'expert_data', args.expert_db)
    expert_state, expert_action, expert_next_state = get_db_mjrl(expert_db_path, args.num_trajs) # Expert DB
    max_expert_norm = torch.max(torch.norm(expert_state, p=2, dim=1))

    # Buffer storing online interactions to iteratively train dynamics models
    online_dataset = OnlineDataset(args.env, args.buffer_size, device=device)

    # ========= Create Model Ensemble =========
    optim_args = {'optim': args.dynamics_optim, 'lr': args.dynamics_lr, 'momentum': args.dynamics_momentum, 'eps':1e-8}
    model_ensemble = DynamicsEnsemble(args.env, args.n_models, online_dataset, hidden_sizes=args.dynamics_model_hidden, \
            optim_args = optim_args, base_seed=args.seed, num_cpus=args.num_cpu)

    # ======== ENV SETUP ========
    logger.info(">>>>> Creating Environments")
    inf_env = GymEnv(gym.make(args.env))
    mb_env = GymEnv(model_based_env(gym.make(args.env), model_ensemble, init_state_buffer=expert_state.numpy(),\
                                    norm_thresh = args.norm_thresh_coeff*max_expert_norm, device=device))

    # ====== Cost Setup =======
    # NOTE: (State) and (State, Next State) specific
    cost_input = expert_state
    input_type = 's'
    if args.use_next_state:
        cost_input = torch.cat([expert_state, expert_next_state], dim=1)
        input_type = 'ss'
    cost_function = RBFLinearCost(cost_input, update_type=args.update_type, feature_dim=args.feature_dim, \
            input_type=input_type, bw_quantile=args.bw_quantile, lambda_b=args.lambda_b, lr=args.cost_lr, seed=args.seed)

    # ============= INIT AGENT =============
    policy = MLP(inf_env.spec, hidden_sizes=tuple(args.actor_model_hidden), seed=args.seed,
                 init_log_std=args.policy_init_log, min_log_std=args.policy_min_log)
    baseline = MLPBaseline(inf_env.spec, reg_coef=args.vf_reg_coef, batch_size=args.vf_batch_size, \
                           hidden_sizes=tuple(args.critic_model_hidden), epochs=args.vf_iters, learn_rate=args.vf_lr)

    # ============== Policy Gradient Init =============
    if args.planner == 'trpo':
        cg_args = {'iters': args.cg_iter, 'damping': args.cg_damping}
        planner_agent = NPG(mb_env, policy, baseline, normalized_step_size=args.kl_dist, \
                    hvp_sample_frac=args.hvp_sample_frac, seed=args.seed, FIM_invert_args=cg_args, save_logs=True)
    elif args.planner == 'ppo':
        planner_agent = PPO(mb_env, policy, baseline, clip_coef=args.clip_coef, epochs=args.ppo_epochs, \
                            mb_size=args.ppo_batch_size, learn_rate=args.ppo_lr, save_logs=True)
    else:
        raise NotImplementedError('Chosen Planner not yet supported')

    # ==============================================
    # ============== MAIN LOOP START ===============
    # ==============================================

    n_iter = 0
    best_policy_score = -float('inf')
    greedy_scores, sample_scores, greedy_mmds, sample_mmds = [], [], [], []
    while n_iter<args.n_iter:
        logger.info(f"{'='*10} Main Episode {n_iter+1} {'='*10}")
        # ============= Evaluate, Save, Plot ===============
        scores, mmds = evaluate(n_iter, logger, writer, args, inf_env, \
                                planner_agent.policy, cost_function, num_traj=20)
        save_and_plot(n_iter, args, dirs, scores, mmds)

        if scores['greedy'] > best_policy_score:
            best_policy_score = scores['greedy']
            save_checkpoint(dirs, planner_agent, cost_function, 'best', agent_type=args.planner)

        if (n_iter+1) % args.save_iter == 0:
            save_checkpoint(dirs, planner_agent, cost_function, n_iter+1, agent_type=args.planner)

        # =============== FIT DYNAMICS ===============
        logger.info('====== Updating Dynamics Model =======')
        if n_iter == 0:
            online_samples = sample_data_batch(args.n_pretrain_samples, inf_env, planner_agent.policy, \
                base_seed=args.seed, num_cpu=args.num_cpu, eval_mode=args.greedy_pretrain)
        else:
            online_samples = sample_data_batch(args.n_dynamics_samples, inf_env, planner_agent.policy, \
                base_seed=args.seed, num_cpu=args.num_cpu, eval_mode=args.greedy_pretrain)
        states = torch.from_numpy(np.concatenate([traj['observations'] for traj in online_samples], axis=0)).float()
        actions = torch.from_numpy(np.concatenate([traj['actions'] for traj in online_samples], axis=0)).float()
        next_states = torch.from_numpy(np.concatenate([traj['next_observations'] for traj in online_samples], axis=0)).float()
        model_ensemble.add_data(states, actions, next_states)
        training_epochs = args.n_pretrain_epochs if n_iter == 0 else args.n_epochs
        model_train_info = model_ensemble.train(training_epochs, logger=logger, grad_clip=args.grad_clip)
        for n, info in enumerate(model_train_info):
            writer.add_scalars(f'data/dynamics_model_{n+1}', {'start_loss': info[1],
                                                              'min_loss': info[0]}, n_iter)
            for n_grad, grad in enumerate(info[2]):
                writer.add_scalar(f'data/grad_norms_{n+1}', grad, n_iter*args.n_epochs + n_grad)

        model_ensemble.compute_threshold()
        logger.info(f">>>>> Computed Maximum Discrepancy for Ensemble: {model_ensemble.threshold}")
        mb_env.env.update_model(model_ensemble)

        # BW update
        #if args.update_bw:
        #    old_bw = cost_function.bw
        #    cost_function.update_bandwidth(model_ensemble.dataset)
        #    logger.info(f">>>>> Updated Bandwidth from {old_bw} to {cost_function.bw}")

        # =============== START MINMAX ==============
        for j in range(args.n_minmax):
            logger.info(f'====== TRPO Step {j} =======')
            best_baseline_optim, best_baseline = None, None
            best_policy, best_prev_policy = None, None
            curr_max_reward, curr_min_vloss = -float('inf'), float('inf')
            for i in range(args.pg_iter):
                reward_kwargs = dict(reward_func=cost_function, ensemble=model_ensemble, device=device, \
                                     i=i, use_ground_truth=args.use_ground_truth, update_bw=args.update_bw, \
                                     use_next_state=args.use_next_state, logger=logger)
                planner_args = dict(N=args.samples_per_step, env=mb_env, sample_mode='model_based', \
                                    gamma=args.gamma, gae_lambda=args.gae_lambda, num_cpu=args.num_cpu, \
                                    reward_kwargs=reward_kwargs)
                prev_policy = planner_agent.policy.get_param_values()
                r_mean, r_std, r_min, r_max, _, infos  = planner_agent.train_step(**planner_args)

                # Policy Heuristic
                if r_mean > curr_max_reward:
                    curr_max_reward = r_mean
                    best_policy = planner_agent.policy.get_param_values()
                    best_prev_policy = prev_policy
                    #best_baseline = planner_agent.baseline.model.state_dict()
                    #best_baseline_optim = planner_agent.baseline.optimizer.state_dict()

                # Baseline Heuristic
                if infos['vf_loss_end'] < curr_min_vloss:
                    curr_min_vloss = infos['vf_loss_end']
                    best_baseline = planner_agent.baseline.model.state_dict()
                    best_baseline_optim = planner_agent.baseline.optimizer.state_dict()

                # Stderr Logging
                reward_mean = np.array(infos['reward']).mean()
                int_mean = np.array(infos['int']).mean()
                ext_mean = np.array(infos['ext']).mean()
                len_mean = np.array(infos['ep_len']).mean()
                ground_truth_mean = np.array(infos['ground_truth_reward']).mean()
                if infos['mb_mmd'] is not None:
                    logger.info(f'Model MMD: {infos["mb_mmd"]}')
                logger.info(f'Bonus MMD: {infos["bonus_mmd"]}')
                logger.info(f'Model Ground Truth Reward: {ground_truth_mean}')
                logger.info('PG Iteration {} reward | int | ext | ep_len ---- {:.2f} | {:.2f} | {:.2f} | {:.2f}' \
                            .format(i+1, reward_mean, int_mean, ext_mean, len_mean))

                # Tensorboard Logging
                step_count = n_iter*args.n_minmax + j*args.pg_iter + i
                if len(infos['int']) != 0:
                    writer.add_scalar('data/reward_mean', reward_mean, step_count)
                    writer.add_scalar('data/ext_reward_mean', ext_mean, step_count)
                    writer.add_scalar('data/int_reward_mean', int_mean, step_count)
                    writer.add_scalar('data/ep_len_mean', len_mean, step_count)
                writer.add_scalar('data/true_reward_mean', ground_truth_mean, step_count)
                writer.add_scalar('data/value_loss', infos['vf_loss_end'], step_count)
                if infos['mb_mmd'] is not None:
                    writer.add_scalar('data/mb_mmd', infos['mb_mmd'], n_iter*args.n_minmax + j)
                if infos['bonus_mmd'] is not None:
                    writer.add_scalar('data/bonus_mmd', infos['bonus_mmd'], step_count)

            # repopulate planner_agent.policy and baseline, optimizer model weights
            planner_agent.policy.set_param_values(best_prev_policy, set_new=False, set_old=True) # Set old
            planner_agent.policy.set_param_values(best_policy, set_new=True, set_old=False) # Set new
            planner_agent.baseline.model.load_state_dict(best_baseline)
            planner_agent.baseline.optimizer.load_state_dict(best_baseline_optim)
        n_iter += 1

if __name__ == '__main__':
    main()
