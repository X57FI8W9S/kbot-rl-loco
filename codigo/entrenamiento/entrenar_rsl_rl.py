#!/usr/bin/env python3
import argparse
import os
import torch
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher
from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop

def main():
    # 1. Parse Arguments & Launch Isaac Sim
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iteraciones", type=int, default=300)
    parser.add_argument(
        "--fabric-cloning",
        action="store_true",
        help="Activa clone_in_fabric en la escena. Dejar apagado evita fallos de clonacion en escenas no compatibles.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Standard imports (MUST be after app_launcher to avoid Isaac Sim crashes)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from entornos.marcha_rsl_rl_env import ConfiguracionEntornoMarchaRslRl, EntornoMarchaRslRl
    

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str((Path(__file__).resolve().parents[2] / "salidas/rsl_rl_marcha_vectorizado"/ marca_tiempo).resolve())
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Environment
    cfg_entorno = ConfiguracionEntornoMarchaRslRl(
        dispositivo=device,
        num_entornos=args.num_envs,
        usar_clone_en_fabric=args.fabric_cloning,
    )
    cfg_robot = ConfiguracionKBotBoxTop()
    
    env_base = EntornoMarchaRslRl(cfg_entorno, cfg_robot, headless=args.headless)
    env = RslRlVecEnvWrapper(env_base) # Standard Isaac Lab wrapper

    # 3. Configure RSL-RL PPO
    
    cfg_rsl = {
        "seed": 42,
        "device": device,
        "num_steps_per_env": 64,
        "save_interval": 25,
        "experiment_name": "marcha_rsl_rl",
        "run_name": marca_tiempo,
        "logger": "tensorboard",
        "resume": False,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.135,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 10,
            "num_mini_batches": 128,
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "rnd_cfg": None,
        },
    }

    # 4. Train!
    print(f"[INFO] Iniciando entrenamiento. Tensorboard logs en: {log_dir}")
    runner = OnPolicyRunner(env, cfg_rsl, log_dir=log_dir, device=device)
    # runner.learn(num_learning_iterations=args.iteraciones, init_at_random_ep_len=False)
    runner.learn(num_learning_iterations=args.iteraciones, init_at_random_ep_len=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
