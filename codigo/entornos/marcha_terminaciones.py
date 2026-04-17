from __future__ import annotations

import torch


def calcular_terminaciones(entorno):
    altura = entorno.robot.data.root_pos_w[:, 2]
    coseno_vertical = torch.abs(entorno.robot.data.projected_gravity_b[:, 2])

    caida_por_altura = altura < entorno.cfg.altura_minima_base
    caida_por_inclinacion = coseno_vertical < entorno.cfg.coseno_minimo_vertical
    truncado_por_tiempo = entorno.episode_length_buf >= entorno.cfg.pasos_maximos_episodio

    terminated = caida_por_altura | caida_por_inclinacion
    truncated = truncado_por_tiempo
    info = {
        "caida_por_altura": caida_por_altura,
        "caida_por_inclinacion": caida_por_inclinacion,
        "truncado_por_tiempo": truncado_por_tiempo,
    }
    return terminated, truncated, info
