#!/usr/bin/env python3
import argparse
import os
import torch
from pathlib import Path

from isaaclab.app import AppLauncher
from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop

def main():
    # 1. Parse Arguments & Launch Isaac Sim
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iteraciones", type=int, default=500)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Standard imports (MUST be after app_launcher to avoid Isaac Sim crashes)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from entornos.marcha_rsl_rl_env import ConfiguracionEntornoMarchaRslRl, EntornoMarchaRslRl

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_dir = str((Path(__file__).resolve().parents[2] / "salidas/rsl_rl_marcha_vectorizado").resolve())
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Environment
    cfg_entorno = ConfiguracionEntornoMarchaRslRl(dispositivo=device, num_entornos=args.num_envs)
    cfg_robot = ConfiguracionKBotBoxTop()
    
    env_base = EntornoMarchaRslRl(cfg_entorno, cfg_robot, headless=args.headless)
    env = RslRlVecEnvWrapper(env_base) # Standard Isaac Lab wrapper

    # 3. Configure RSL-RL PPO
    cfg_rsl = {
        "seed": 42,
        "runner": {
            "num_steps_per_env": 256,
            "max_iterations": args.iteraciones,
            "save_interval": 25,
            "experiment_name": "marcha_rsl_rl",
            "run_name": "",
            "logger": "tensorboard",
            "resume": False,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.135,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": max(1, (args.num_envs * 256) // 4096),
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
        },
    }

    # 4. Train!
    print(f"[INFO] Iniciando entrenamiento. Tensorboard logs en: {log_dir}")
    runner = OnPolicyRunner(env, cfg_rsl, log_dir=log_dir, device=device)
    runner.learn(num_learning_iterations=args.iteraciones, init_at_random_ep_len=False)

    simulation_app.close()

if __name__ == "__main__":
    main()