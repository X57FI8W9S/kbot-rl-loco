#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

from isaaclab.app import AppLauncher


class RedActorCritico(nn.Module):
    def __init__(self, dim_obs: int, dim_accion: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(dim_obs, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, dim_accion),
        )
        self.critic = nn.Sequential(
            nn.Linear(dim_obs, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        self.log_std = nn.Parameter(torch.full((dim_accion,), -2.0))

    def accion_media(self, observacion: torch.Tensor) -> torch.Tensor:
        return self.actor(observacion)


def resolver_checkpoint(ruta_checkpoint: str | None) -> Path:
    raiz_repo = Path(__file__).resolve().parents[2]

    if ruta_checkpoint:
        ruta_entrada = Path(ruta_checkpoint).expanduser()
        ruta = ruta_entrada if ruta_entrada.is_absolute() else (raiz_repo / ruta_entrada)
        ruta = ruta.resolve()
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el checkpoint: {ruta}")
        return ruta

    directorio = raiz_repo / "salidas" / "ppo_marcha_vectorizado"
    candidatos = sorted(directorio.glob("modelo_*.pt"))
    if not candidatos:
        raise FileNotFoundError(f"No encontre checkpoints en {directorio}")
    return candidatos[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Ruta al checkpoint .pt. Si se omite, usa el mas reciente.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-pasos", type=int, default=4000)
    parser.add_argument("--espaciado-entornos", type=float, default=2.5)
    parser.add_argument("--sample", action="store_true", help="Muestrea acciones con ruido en lugar de usar la media.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completado = False

    try:
        from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop
        from entornos.marcha_ppo_env import ConfiguracionEntornoMarchaPPO, EntornoMarchaPPO

        ruta_checkpoint = resolver_checkpoint(args.checkpoint)
        print(f"[INFO] Cargando checkpoint: {ruta_checkpoint}", flush=True)

        contenido = torch.load(ruta_checkpoint, map_location=args.device)
        dim_obs = int(contenido["obs_dim"])
        dim_accion = int(contenido["act_dim"])

        modelo = RedActorCritico(dim_obs, dim_accion).to(args.device)
        modelo.load_state_dict(contenido["modelo"])
        modelo.eval()

        cfg_entorno = ConfiguracionEntornoMarchaPPO(
            dispositivo=args.device,
        )
        entorno = EntornoMarchaPPO(
            configuracion_entorno=cfg_entorno,
            configuracion_robot=ConfiguracionKBotBoxTop(),
            headless=args.headless,
        )

        observacion = entorno.reiniciar().to(args.device)
        for paso in range(args.num_pasos):
            with torch.no_grad():
                media = modelo.accion_media(observacion)
                if args.sample:
                    std = torch.exp(modelo.log_std).expand_as(media)
                    accion = torch.normal(mean=media, std=std)
                else:
                    accion = media
                accion = accion.clamp(-1.0, 1.0)

            observacion, recompensa, terminado, _ = entorno.paso(accion)
            observacion = observacion.to(args.device)

            if paso % 200 == 0:
                print(
                    f"[PASO {paso:05d}] "
                    f"recompensa_media={float(recompensa.mean().item()): .4f} "
                    f"terminaciones={int(terminado.sum().item())} "
                    f"cmd_x={float(entorno.velocidad_objetivo_x.mean().item()): .3f}",
                    flush=True,
                )

        completado = True

    except Exception as exc:
        print(f"[ERROR] Fallo al visualizar checkpoint: {exc}", flush=True)
        traceback.print_exc()
        raise

    finally:
        if completado:
            print("[INFO] Visualizacion finalizada.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        print("[INFO] Cerrando Isaac Sim tras interrupcion o error...", flush=True)
        simulation_app.close()
        print("[INFO] Isaac Sim cerrado.", flush=True)


if __name__ == "__main__":
    main()
