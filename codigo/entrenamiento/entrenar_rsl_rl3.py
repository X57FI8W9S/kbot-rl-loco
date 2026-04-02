#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
import signal
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher
from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop


@dataclass
class ConfiguracionEntrenamientoRslRl:
    dispositivo: str = "cuda:0"
    num_entornos: int = 64
    pasos_rollout: int = 256
    iteraciones: int = 500
    epocas_optimizacion: int = 5
    tamano_minibatch: int = 4096

    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    coef_valor: float = 1.0
    coef_entropia: float = 0.01
    norma_gradiente_max: float = 1.0
    lr: float = 3e-4

    guardar_cada: int = 25
    directorio_salida: str = "salidas/rsl_rl_marcha_vectorizado"
    experimento: str = "marcha_rsl_rl"
    nombre_run: str = ""


def resolver_dispositivo_preferido(dispositivo_solicitado: str | None) -> str:
    if dispositivo_solicitado:
        if dispositivo_solicitado.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Se solicito '{dispositivo_solicitado}', pero torch.cuda.is_available() es False."
            )
        return dispositivo_solicitado
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolver_directorio_salida(directorio_salida: str) -> str:
    raiz_repo = Path(__file__).resolve().parents[2]
    return str((raiz_repo / directorio_salida).resolve())


def nombre_senal(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIGNAL_{signum}"


def importar_wrapper_rsl_rl():
    candidatos = [
        ("isaaclab_rl.rsl_rl", "RslRlVecEnvWrapper"),
        ("isaaclab_tasks.utils.wrappers.rsl_rl", "RslRlVecEnvWrapper"),
        ("omni.isaac.lab_tasks.utils.wrappers.rsl_rl", "RslRlVecEnvWrapper"),
    ]
    errores = []
    for modulo, nombre in candidatos:
        try:
            mod = __import__(modulo, fromlist=[nombre])
            return getattr(mod, nombre)
        except Exception as exc:
            errores.append(f"{modulo}.{nombre}: {exc}")
    raise ImportError("No pude importar RslRlVecEnvWrapper.\n" + "\n".join(errores))


class AdaptadorVecEnvRslRlNativo:
    """Adapter para usar el entorno directo cuando el wrapper de Isaac Lab no esta disponible."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.num_actions = env.num_actions
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs

    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"]

    def get_observations(self):
        return self.env.get_observations()

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        return obs_dict["policy"], rewards, dones, extras


def construir_cfg_rsl_rl(cfg: ConfiguracionEntrenamientoRslRl) -> dict[str, Any]:
    mini_batches = max(1, (cfg.num_entornos * cfg.pasos_rollout) // cfg.tamano_minibatch)
    return {
        "runner_class_name": "OnPolicyRunner",
        "seed": 42,
        "device": cfg.dispositivo,
        "runner": {
            "num_steps_per_env": cfg.pasos_rollout,
            "max_iterations": cfg.iteraciones,
            "save_interval": cfg.guardar_cada,
            "experiment_name": cfg.experimento,
            "run_name": cfg.nombre_run,
            "logger": "tensorboard",
            "resume": False,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.1353352832366127,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": cfg.coef_valor,
            "use_clipped_value_loss": True,
            "clip_param": cfg.clip_epsilon,
            "entropy_coef": cfg.coef_entropia,
            "num_learning_epochs": cfg.epocas_optimizacion,
            "num_mini_batches": mini_batches,
            "learning_rate": cfg.lr,
            "schedule": "fixed",
            "gamma": cfg.gamma,
            "lam": cfg.lambda_gae,
            "desired_kl": 0.01,
            "max_grad_norm": cfg.norma_gradiente_max,
            "normalize_advantage_per_mini_batch": False,
        },
    }


def aplanar_cfg_runner(cfg_rsl: dict[str, Any]) -> dict[str, Any]:
    runner_cfg = dict(cfg_rsl["runner"])
    runner_cfg["algorithm"] = dict(cfg_rsl["algorithm"])

    if "policy" in cfg_rsl:
        policy_cfg = dict(cfg_rsl["policy"])
        if "actor_hidden_dims" in policy_cfg or "critic_hidden_dims" in policy_cfg:
            runner_cfg["actor"] = {
                "class_name": "MLPModel",
                "hidden_dims": list(policy_cfg.get("actor_hidden_dims", [256, 256])),
                "activation": policy_cfg.get("activation", "elu"),
                "obs_normalization": False,
                "stochastic": True,
                "init_noise_std": policy_cfg.get("init_noise_std", 1.0),
                "noise_std_type": "scalar",
                "state_dependent_std": False,
            }
            runner_cfg["critic"] = {
                "class_name": "MLPModel",
                "hidden_dims": list(policy_cfg.get("critic_hidden_dims", [256, 256])),
                "activation": policy_cfg.get("activation", "elu"),
                "obs_normalization": False,
                "stochastic": False,
            }
        else:
            runner_cfg["policy"] = policy_cfg

    return runner_cfg


def crear_runner(on_policy_runner_cls, env_entrenamiento, cfg_rsl: dict[str, Any], log_dir: str, device: str):
    errores = []
    for train_cfg in (cfg_rsl, aplanar_cfg_runner(cfg_rsl)):
        try:
            return on_policy_runner_cls(env_entrenamiento, train_cfg, log_dir=log_dir, device=device)
        except Exception as exc:
            errores.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("No pude construir OnPolicyRunner con ninguna variante de configuracion.\n" + "\n".join(errores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iteraciones", type=int, default=500)
    parser.add_argument("--pasos-rollout", type=int, default=256)
    parser.add_argument("--espaciado-entornos", type=float, default=2.5)
    parser.add_argument("--log-dir", type=str, default="salidas/rsl_rl_marcha_vectorizado")
    parser.add_argument("--run-name", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completado = False
    motivo_salida = None
    codigo_salida = 1
    manejadores_previos: dict[int, object] = {}

    def manejar_senal(signum, _frame):
        nonlocal motivo_salida
        motivo_salida = f"senal recibida: {nombre_senal(signum)} ({signum})"
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if hasattr(signal, signal.Signals(signum).name):
            manejadores_previos[signum] = signal.getsignal(signum)
            signal.signal(signum, manejar_senal)

    try:
        from rsl_rl.runners import OnPolicyRunner

        from entornos.marcha_rsl_rl_env import ConfiguracionEntornoMarchaRslRl, EntornoMarchaRslRl

        dispositivo = resolver_dispositivo_preferido(args.device)
        cfg_train = ConfiguracionEntrenamientoRslRl(
            dispositivo=dispositivo,
            num_entornos=args.num_envs,
            iteraciones=args.iteraciones,
            pasos_rollout=args.pasos_rollout,
            directorio_salida=resolver_directorio_salida(args.log_dir),
            nombre_run=args.run_name,
        )

        cfg_entorno = ConfiguracionEntornoMarchaRslRl(
            dispositivo=cfg_train.dispositivo,
            num_entornos=cfg_train.num_entornos,
            espaciado_entornos=args.espaciado_entornos,
        )
        cfg_robot = ConfiguracionKBotBoxTop()

        env_base = EntornoMarchaRslRl(cfg_entorno, cfg_robot, headless=args.headless)

        try:
            RslRlVecEnvWrapper = importar_wrapper_rsl_rl()
            env_entrenamiento = RslRlVecEnvWrapper(env_base)
            usando_wrapper = True
        except ImportError:
            env_entrenamiento = AdaptadorVecEnvRslRlNativo(env_base)
            usando_wrapper = False

        os.makedirs(cfg_train.directorio_salida, exist_ok=True)
        cfg_rsl = construir_cfg_rsl_rl(cfg_train)
        runner = crear_runner(
            on_policy_runner_cls=OnPolicyRunner,
            env_entrenamiento=env_entrenamiento,
            cfg_rsl=cfg_rsl,
            log_dir=cfg_train.directorio_salida,
            device=cfg_train.dispositivo,
        )

        if hasattr(runner, "add_git_repo_to_log"):
            runner.add_git_repo_to_log(__file__)

        print(
            "[INFO] Iniciando entrenamiento rsl_rl "
            f"(wrapper_isaaclab={'si' if usando_wrapper else 'no'}, "
            f"num_envs={cfg_train.num_entornos}, "
            f"num_steps_per_env={cfg_train.pasos_rollout}, "
            f"iteraciones={cfg_train.iteraciones}, "
            f"log_dir={cfg_train.directorio_salida})",
            flush=True,
        )

        if "logger" in inspect.signature(runner.learn).parameters:
            runner.learn(num_learning_iterations=cfg_train.iteraciones, init_at_random_ep_len=False)
        else:
            runner.learn(num_learning_iterations=cfg_train.iteraciones, init_at_random_ep_len=False)

        completado = True
        codigo_salida = 0

    except KeyboardInterrupt:
        if motivo_salida is None:
            motivo_salida = "KeyboardInterrupt"
        codigo_salida = 130
        print(f"[ERROR] Entrenamiento interrumpido: {motivo_salida}", file=sys.stderr, flush=True)
        traceback.print_exc()

    except SystemExit as exc:
        if motivo_salida is None:
            motivo_salida = f"SystemExit({exc.code})"
        codigo_salida = exc.code if isinstance(exc.code, int) else 1
        print(f"[ERROR] Salida anticipada: {motivo_salida}", file=sys.stderr, flush=True)

    except Exception as exc:
        motivo_salida = f"{type(exc).__name__}: {exc}"
        codigo_salida = 1
        print(f"[ERROR] Excepcion no controlada: {motivo_salida}", file=sys.stderr, flush=True)
        traceback.print_exc()

    finally:
        for signum, manejador_previo in manejadores_previos.items():
            signal.signal(signum, manejador_previo)

        if completado:
            print("[INFO] Entrenamiento finalizado. Saliendo sin cierre bloqueante de Isaac Sim.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        print(
            f"[INFO] Cerrando Isaac Sim tras interrupcion o error. Motivo: {motivo_salida or 'desconocido'}",
            flush=True,
        )
        simulation_app.close()
        print("[INFO] Isaac Sim cerrado.", flush=True)
        sys.exit(codigo_salida)


if __name__ == "__main__":
    main()
