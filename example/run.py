from argparse import ArgumentParser
parser = ArgumentParser(description='')
parser.add_argument('env_name', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('run_mode', choices=['train', 'play', 'test'])
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--terrain-level', action='store_true')

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ENV_NAME = args_cli.env_name
RUN_PATH = 'example/runs/%s'%args_cli.model_name
MODEL_PATH = 'example/models/%s.pt'%args_cli.model_name
RUN_MODE = args_cli.run_mode
STOCHASTIC = args_cli.stochastic
TERRAIN_LEVEL = args_cli.terrain_level

from simple_rl.runner import PPORunner
from simple_rl.algorithms.ppo import PPO, PPOCfg
from simple_rl.modules.modules import MlpActorCritic

from env_wrapper import LeggedEnvWrapper
import legged_lab.task
import torch as th


def main() :
    env = legged_lab.task.make(ENV_NAME)
    env = LeggedEnvWrapper(env, reward_scale=1/50)
    
    actor_critic = MlpActorCritic(
        n_obs=env.n_obs,
        n_action=env.n_action,
        init_std=1.0,
        net_arch=[512, 256, 128],
        activ_fn=th.nn.ELU
    )
    ppo_cfg = PPOCfg(
        n_rollout=50,
        n_epoch=5,
        n_minibatch=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        desired_kl=0.01,
        normalize_observation=False,
        ratio_clip_param=0.2,
        value_clip_param=0.2,
        grad_norm_clip=1.0,
        normalize_advantage=True,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.0,
    )
    ppo = PPO(env.spec, actor_critic, ppo_cfg)
    
    # train
    if RUN_MODE == 'train' :
        runner = PPORunner(
            env=env,
            algo=ppo,
            run_dir=RUN_PATH,
            log_interval=2,
            checkpoint_interval=300,
        )
        runner.train(3000)
        ppo.save(MODEL_PATH)
    
    # test/play
    elif RUN_MODE == 'test' or RUN_MODE == 'play' :
        if RUN_MODE == 'play' :
            ppo.load(MODEL_PATH)
        
        obs, info = env.reset()
        step_cnt = 0
        
        while simulation_app.is_running() :
            obs, rwd, ter, tru, info = env.step(
                ppo.act(obs, deterministic=(not STOCHASTIC))
            )

            if TERRAIN_LEVEL and (step_cnt % 10 == 0):
                x_grid_num = env.env.command_manager.cfg.x_grid_num
                y_grid_num = env.env.command_manager.cfg.y_grid_num
                n_grid = env.env.command_manager.n_grid
                grid_ids = env.env.command_manager._grid_ids
                terrain_levels = env.env.scene.terrain.terrain_levels

                grid_level_sum = th.bincount(grid_ids, terrain_levels, n_grid)
                grid_level_cnt = th.bincount(grid_ids, minlength=n_grid).clip(min=1)
                grid_level = grid_level_sum / grid_level_cnt
                grid_level = grid_level.view(x_grid_num, y_grid_num)
                
                output = []
                output.append('\n\n\n\n')
                output.append('global terrain level : %5.3f\n'%terrain_levels.to(th.float32).mean().item())
                output.append('grid terrain level :\n')
                for rows in reversed(grid_level.tolist()) :
                    output.extend(['%4.2f  '% ele for ele in rows])
                    output.append('\n')
                output = ''.join(output)
                print(output)
            
            step_cnt += 1
    
    env.close()


if __name__ == '__main__' :
    main()
    simulation_app.close()
