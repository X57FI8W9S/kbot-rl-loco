#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab.app import AppLauncher
from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop

CHECKPOINT = "salidas/rsl_rl_marcha_vectorizado/model_75.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-pasos", type=int, default=4000)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from entornos.marcha_rsl_rl_env import ConfiguracionEntornoMarchaRslRl, EntornoMarchaRslRl

    raiz_repo = Path(__file__).resolve().parents[2]
    ruta_checkpoint = (raiz_repo / CHECKPOINT).resolve()
    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {ruta_checkpoint}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg_entorno = ConfiguracionEntornoMarchaRslRl(dispositivo=device, num_entornos=args.num_envs)
    env_base = EntornoMarchaRslRl(cfg_entorno, ConfiguracionKBotBoxTop(), headless=args.headless)
    env = RslRlVecEnvWrapper(env_base)

    pasos_rollout = 256
    cfg_rsl = {
        "seed": 42,
        "device": device,
        "num_steps_per_env": pasos_rollout,
        "save_interval": 25,
        "experiment_name": "marcha_rsl_rl",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
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
            "num_learning_epochs": 5,
            "num_mini_batches": max(1, (args.num_envs * pasos_rollout) // 4096),
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "rnd_cfg": None,
        },
    }

    print(f"[INFO] Cargando checkpoint: {ruta_checkpoint}", flush=True)
    runner = OnPolicyRunner(env, cfg_rsl, log_dir=None, device=device)
    runner.load(str(ruta_checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    for paso in range(args.num_pasos):
        with torch.inference_mode():
            acciones = policy(obs)
            obs, recompensas, dones, extras = env.step(acciones)

        if paso % 200 == 0:
            print(
                f"[PASO {paso:05d}] "
                f"recompensa_media={float(recompensas.mean().item()): .4f} "
                f"terminaciones={int(dones.sum().item())}",
                flush=True,
            )
            if extras and "log" in extras and "episode" in extras["log"]:
                print(f"[INFO] episode={extras['log']['episode']}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
