#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from isaaclab.app import AppLauncher
from configuraciones.kbot_box_top import ConfiguracionKBotBoxTop


@dataclass
class ConfiguracionPPOVectorizado:
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
    directorio_salida: str = "salidas/ppo_marcha_vectorizado"


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

    def actuar(self, observacion: torch.Tensor):
        media = self.actor(observacion)
        std = torch.exp(self.log_std).expand_as(media)
        distribucion = Normal(media, std)
        accion = distribucion.sample()
        log_prob = distribucion.log_prob(accion).sum(dim=-1)
        valor = self.critic(observacion).squeeze(-1)
        return accion, log_prob, valor

    def evaluar(self, observacion: torch.Tensor, accion: torch.Tensor):
        media = self.actor(observacion)
        std = torch.exp(self.log_std).expand_as(media)
        distribucion = Normal(media, std)
        log_prob = distribucion.log_prob(accion).sum(dim=-1)
        entropia = distribucion.entropy().sum(dim=-1)
        valor = self.critic(observacion).squeeze(-1)
        return log_prob, entropia, valor


def calcular_gae_vectorizado(
    recompensas: torch.Tensor,
    dones: torch.Tensor,
    valores: torch.Tensor,
    valor_final: torch.Tensor,
    gamma: float,
    lambda_gae: float,
):
    ventajas = torch.zeros_like(recompensas)
    gae = torch.zeros_like(valor_final)
    for t in reversed(range(recompensas.shape[0])):
        mascara = 1.0 - dones[t].float()
        siguiente_valor = valor_final if t == recompensas.shape[0] - 1 else valores[t + 1]
        delta = recompensas[t] + gamma * siguiente_valor * mascara - valores[t]
        gae = delta + gamma * lambda_gae * mascara * gae
        ventajas[t] = gae
    retornos = ventajas + valores
    return ventajas, retornos


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iteraciones", type=int, default=500)
    parser.add_argument("--pasos-rollout", type=int, default=256)
    parser.add_argument("--espaciado-entornos", type=float, default=2.5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    completado = False

    try:
        from entornos.marcha_ppo_vec_env import (
            ConfiguracionEntornoMarchaPPOVec,
            EntornoMarchaPPOVec,
        )

        dispositivo = resolver_dispositivo_preferido(args.device)
        cfg_ppo = ConfiguracionPPOVectorizado(
            dispositivo=dispositivo,
            num_entornos=args.num_envs,
            iteraciones=args.iteraciones,
            pasos_rollout=args.pasos_rollout,
        )
        cfg_ppo.directorio_salida = resolver_directorio_salida(cfg_ppo.directorio_salida)

        cfg_entorno = ConfiguracionEntornoMarchaPPOVec(
            dispositivo=cfg_ppo.dispositivo,
            num_entornos=cfg_ppo.num_entornos,
            espaciado_entornos=args.espaciado_entornos,
        )
        cfg_robot = ConfiguracionKBotBoxTop()

        entorno = EntornoMarchaPPOVec(cfg_entorno, cfg_robot, headless=args.headless)
        modelo = RedActorCritico(entorno.obs_dim, entorno.act_dim).to(cfg_ppo.dispositivo)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg_ppo.lr)

        observacion = entorno.reiniciar().to(cfg_ppo.dispositivo)

        for iteracion in range(cfg_ppo.iteraciones):
            buffer_obs = []
            buffer_acciones = []
            buffer_logprob = []
            buffer_recompensas = []
            buffer_dones = []
            buffer_valores = []

            for _ in range(cfg_ppo.pasos_rollout):
                with torch.no_grad():
                    accion, log_prob, valor = modelo.actuar(observacion)

                siguiente_obs, recompensa, done, _ = entorno.paso(accion)

                buffer_obs.append(observacion)
                buffer_acciones.append(accion)
                buffer_logprob.append(log_prob)
                buffer_recompensas.append(recompensa)
                buffer_dones.append(done)
                buffer_valores.append(valor)

                observacion = siguiente_obs.to(cfg_ppo.dispositivo)

            with torch.no_grad():
                _, _, valor_final = modelo.actuar(observacion)

            observaciones = torch.stack(buffer_obs)
            acciones = torch.stack(buffer_acciones)
            logprob_viejo = torch.stack(buffer_logprob)
            recompensas = torch.stack(buffer_recompensas)
            dones = torch.stack(buffer_dones)
            valores = torch.stack(buffer_valores)

            ventajas, retornos = calcular_gae_vectorizado(
                recompensas=recompensas,
                dones=dones,
                valores=valores,
                valor_final=valor_final,
                gamma=cfg_ppo.gamma,
                lambda_gae=cfg_ppo.lambda_gae,
            )

            ventajas = (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)

            observaciones = observaciones.reshape(-1, entorno.obs_dim)
            acciones = acciones.reshape(-1, entorno.act_dim)
            logprob_viejo = logprob_viejo.reshape(-1)
            ventajas = ventajas.reshape(-1)
            retornos = retornos.reshape(-1)

            cantidad_muestras = observaciones.shape[0]
            indices = torch.arange(cantidad_muestras, device=cfg_ppo.dispositivo)

            for _ in range(cfg_ppo.epocas_optimizacion):
                indices = indices[torch.randperm(cantidad_muestras, device=cfg_ppo.dispositivo)]
                for inicio in range(0, cantidad_muestras, cfg_ppo.tamano_minibatch):
                    fin = inicio + cfg_ppo.tamano_minibatch
                    idx = indices[inicio:fin]

                    obs_mb = observaciones[idx]
                    act_mb = acciones[idx]
                    logprob_viejo_mb = logprob_viejo[idx]
                    ventajas_mb = ventajas[idx]
                    retornos_mb = retornos[idx]

                    logprob_nuevo, entropia, valor_nuevo = modelo.evaluar(obs_mb, act_mb)

                    ratio = torch.exp(logprob_nuevo - logprob_viejo_mb)
                    objetivo1 = ratio * ventajas_mb
                    objetivo2 = torch.clamp(
                        ratio,
                        1.0 - cfg_ppo.clip_epsilon,
                        1.0 + cfg_ppo.clip_epsilon,
                    ) * ventajas_mb

                    perdida_actor = -torch.min(objetivo1, objetivo2).mean()
                    perdida_critic = 0.5 * ((retornos_mb - valor_nuevo) ** 2).mean()
                    perdida_entropia = entropia.mean()

                    perdida_total = (
                        perdida_actor
                        + cfg_ppo.coef_valor * perdida_critic
                        - cfg_ppo.coef_entropia * perdida_entropia
                    )

                    optimizador.zero_grad()
                    perdida_total.backward()
                    nn.utils.clip_grad_norm_(modelo.parameters(), cfg_ppo.norma_gradiente_max)
                    optimizador.step()

            recompensa_media = float(recompensas.mean().item())
            valor_medio = float(valores.mean().item())
            episodios_terminados = int(dones.sum().item())
            print(
                f"[ITER {iteracion:04d}] "
                f"num_envs={cfg_ppo.num_entornos} "
                f"recompensa_media={recompensa_media: .4f} "
                f"valor_medio={valor_medio: .4f} "
                f"cmd_x={float(entorno.velocidad_objetivo_x.mean().item()): .3f} "
                f"terminaciones={episodios_terminados}"
            )

            if iteracion % cfg_ppo.guardar_cada == 0:
                ruta = f"{cfg_ppo.directorio_salida}/modelo_{iteracion:04d}.pt"
                os.makedirs(cfg_ppo.directorio_salida, exist_ok=True)
                torch.save(
                    {
                        "modelo": modelo.state_dict(),
                        "obs_dim": entorno.obs_dim,
                        "act_dim": entorno.act_dim,
                        "num_envs": cfg_ppo.num_entornos,
                        "nombres_articulaciones_controladas": [
                            entorno.nombres_articulaciones[i] for i in entorno.indices_controlados.tolist()
                        ],
                    },
                    ruta,
                )
                print(f"[INFO] Guardado: {ruta}")

        completado = True

    finally:
        if completado:
            print("[INFO] Entrenamiento finalizado. Saliendo sin cierre bloqueante de Isaac Sim.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        print("[INFO] Cerrando Isaac Sim tras interrupcion o error...", flush=True)
        simulation_app.close()
        print("[INFO] Isaac Sim cerrado.", flush=True)


if __name__ == "__main__":
    main()
