#!/usr/bin/env python3
"""
Componentes base para una política de caminata propioceptiva simple.

Este archivo no intenta ser una tarea completa de Isaac Lab por sí solo.
La idea es separar, en un módulo fácil de reutilizar, la lógica mínima de:
- observación
- recompensa
- terminación
- muestreo de comandos

Así luego se puede conectar a una tarea/env de Isaac Lab sin reescribir todo.

Convenciones:
- nombres de variables y comentarios en español
- operaciones vectorizadas con PyTorch
- pensado para lotes: dimensión [num_entornos, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class ConfiguracionCaminarBasico:
    """Parámetros mínimos para una política de caminata ciega en terreno plano."""

    num_articulaciones_controladas: int
    escala_velocidad_angular_base: float = 0.25
    escala_velocidad_articulaciones: float = 0.05
    escala_accion: float = 0.25
    velocidad_objetivo_min: float = -0.6
    velocidad_objetivo_max: float = 0.8
    altura_minima_base: float = 0.45
    coseno_minimo_vertical: float = 0.55
    penalizacion_caida: float = -5.0
    peso_seguimiento_velocidad: float = 2.0
    peso_verticalidad: float = 0.5
    peso_supervivencia: float = 0.2
    peso_torque: float = 0.00002
    peso_potencia: float = 0.00002
    peso_variacion_accion: float = 0.01
    peso_deslizamiento_pies: float = 0.2
    peso_colision_cuerpo: float = 1.0
    velocidad_sigma: float = 0.25
    pose_nominal_articulaciones: Tensor | None = None

    def validar(self) -> None:
        if self.num_articulaciones_controladas <= 0:
            raise ValueError("num_articulaciones_controladas debe ser mayor que cero")
        if self.pose_nominal_articulaciones is not None:
            if self.pose_nominal_articulaciones.numel() != self.num_articulaciones_controladas:
                raise ValueError(
                    "pose_nominal_articulaciones debe tener exactamente "
                    f"{self.num_articulaciones_controladas} valores"
                )


@dataclass
class EstadoRobot:
    """Contenedor mínimo del estado necesario para observaciones y recompensas."""

    gravedad_proyectada: Tensor
    velocidad_angular_base: Tensor
    posicion_articulaciones: Tensor
    velocidad_articulaciones: Tensor
    accion_anterior: Tensor
    altura_base: Tensor
    velocidad_lineal_base_cuerpo: Tensor
    torques_articulaciones: Tensor
    fuerzas_contacto_cuerpo: Tensor | None = None
    velocidades_pies_xy: Tensor | None = None


def _pose_nominal_expandida(configuracion: ConfiguracionCaminarBasico, num_entornos: int, dispositivo: torch.device) -> Tensor:
    """Devuelve la pose nominal expandida al tamaño del lote."""

    if configuracion.pose_nominal_articulaciones is None:
        return torch.zeros(
            (num_entornos, configuracion.num_articulaciones_controladas),
            dtype=torch.float32,
            device=dispositivo,
        )

    pose_nominal = configuracion.pose_nominal_articulaciones.to(device=dispositivo, dtype=torch.float32)
    return pose_nominal.unsqueeze(0).repeat(num_entornos, 1)


def construir_observacion(
    estado: EstadoRobot,
    velocidad_objetivo_x: Tensor,
    configuracion: ConfiguracionCaminarBasico,
) -> Tensor:
    """
    Construye la observación de una política propioceptiva básica.

    Estructura:
    [gravedad_proyectada(3),
     velocidad_angular_base(3),
     velocidad_objetivo_x(1),
     error_posicion_articulaciones(n),
     velocidad_articulaciones(n),
     accion_anterior(n)]
    """

    num_entornos = estado.posicion_articulaciones.shape[0]
    pose_nominal = _pose_nominal_expandida(configuracion, num_entornos, estado.posicion_articulaciones.device)

    error_posicion_articulaciones = estado.posicion_articulaciones - pose_nominal

    bloque_velocidad_objetivo = velocidad_objetivo_x.view(num_entornos, 1)

    observacion = torch.cat(
        (
            estado.gravedad_proyectada,
            estado.velocidad_angular_base * configuracion.escala_velocidad_angular_base,
            bloque_velocidad_objetivo,
            error_posicion_articulaciones,
            estado.velocidad_articulaciones * configuracion.escala_velocidad_articulaciones,
            estado.accion_anterior,
        ),
        dim=1,
    )
    return observacion


def crear_objetivo_posicion_articulaciones(
    accion_politica: Tensor,
    configuracion: ConfiguracionCaminarBasico,
) -> Tensor:
    """
    Convierte la acción de la política en objetivo de posición articular.

    Supuesto:
    - la política produce acciones normalizadas aproximadamente en [-1, 1]
    - esas acciones son desplazamientos respecto de la pose nominal
    """

    num_entornos = accion_politica.shape[0]
    pose_nominal = _pose_nominal_expandida(configuracion, num_entornos, accion_politica.device)
    return pose_nominal + configuracion.escala_accion * accion_politica


def muestrear_velocidad_objetivo_x(
    num_entornos: int,
    configuracion: ConfiguracionCaminarBasico,
    dispositivo: torch.device,
) -> Tensor:
    """Muestrea una consigna simple de caminar hacia adelante/atrás."""

    return torch.empty(num_entornos, device=dispositivo).uniform_(
        configuracion.velocidad_objetivo_min,
        configuracion.velocidad_objetivo_max,
    )


def _recompensa_seguimiento_velocidad(
    estado: EstadoRobot,
    velocidad_objetivo_x: Tensor,
    configuracion: ConfiguracionCaminarBasico,
) -> Tensor:
    """Premia seguir la velocidad objetivo longitudinal en el marco del cuerpo."""

    error_velocidad = estado.velocidad_lineal_base_cuerpo[:, 0] - velocidad_objetivo_x
    return torch.exp(-(error_velocidad ** 2) / configuracion.velocidad_sigma)


def _recompensa_verticalidad(estado: EstadoRobot) -> Tensor:
    """
    Premia mantenerse erguido.

    Se asume que la gravedad proyectada tiene componente z cercana a -1 o +1,
    según la convención usada. Para no depender del signo, usamos valor absoluto.
    """

    coseno_vertical = torch.abs(estado.gravedad_proyectada[:, 2]).clamp(0.0, 1.0)
    return coseno_vertical


def _penalizacion_torque(estado: EstadoRobot) -> Tensor:
    """Penaliza usar torques altos."""

    return torch.sum(estado.torques_articulaciones ** 2, dim=1)


def _penalizacion_potencia(estado: EstadoRobot) -> Tensor:
    """Penaliza potencia mecánica aproximada: |tau * qd|."""

    return torch.sum(torch.abs(estado.torques_articulaciones * estado.velocidad_articulaciones), dim=1)


def _penalizacion_variacion_accion(accion_actual: Tensor, accion_anterior: Tensor) -> Tensor:
    """Penaliza cambios bruscos entre acciones consecutivas."""

    return torch.sum((accion_actual - accion_anterior) ** 2, dim=1)


def _penalizacion_deslizamiento_pies(estado: EstadoRobot) -> Tensor:
    """
    Penaliza velocidad horizontal de los pies cuando esa señal está disponible.

    Convención esperada de velocidades_pies_xy:
    [num_entornos, num_pies, 2]
    """

    if estado.velocidades_pies_xy is None:
        return torch.zeros(estado.posicion_articulaciones.shape[0], device=estado.posicion_articulaciones.device)

    return torch.sum(estado.velocidades_pies_xy ** 2, dim=(1, 2))


def _penalizacion_colision_cuerpo(estado: EstadoRobot) -> Tensor:
    """
    Penaliza contactos indeseados del cuerpo fuera de los pies.

    Convención esperada de fuerzas_contacto_cuerpo:
    [num_entornos, num_cuerpos_observados, 3]
    """

    if estado.fuerzas_contacto_cuerpo is None:
        return torch.zeros(estado.posicion_articulaciones.shape[0], device=estado.posicion_articulaciones.device)

    magnitud_fuerza = torch.linalg.norm(estado.fuerzas_contacto_cuerpo, dim=2)
    hubo_contacto = (magnitud_fuerza > 1.0).float()
    return torch.sum(hubo_contacto, dim=1)


def calcular_recompensa(
    estado: EstadoRobot,
    accion_actual: Tensor,
    velocidad_objetivo_x: Tensor,
    configuracion: ConfiguracionCaminarBasico,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Calcula recompensa total y sus componentes para logging."""

    recompensa_velocidad = _recompensa_seguimiento_velocidad(estado, velocidad_objetivo_x, configuracion)
    recompensa_verticalidad = _recompensa_verticalidad(estado)
    recompensa_supervivencia = torch.ones_like(recompensa_velocidad)

    penalizacion_torque = _penalizacion_torque(estado)
    penalizacion_potencia = _penalizacion_potencia(estado)
    penalizacion_variacion = _penalizacion_variacion_accion(accion_actual, estado.accion_anterior)
    penalizacion_deslizamiento = _penalizacion_deslizamiento_pies(estado)
    penalizacion_colision = _penalizacion_colision_cuerpo(estado)

    recompensa_total = (
        configuracion.peso_seguimiento_velocidad * recompensa_velocidad
        + configuracion.peso_verticalidad * recompensa_verticalidad
        + configuracion.peso_supervivencia * recompensa_supervivencia
        - configuracion.peso_torque * penalizacion_torque
        - configuracion.peso_potencia * penalizacion_potencia
        - configuracion.peso_variacion_accion * penalizacion_variacion
        - configuracion.peso_deslizamiento_pies * penalizacion_deslizamiento
        - configuracion.peso_colision_cuerpo * penalizacion_colision
    )

    componentes = {
        "recompensa_velocidad": recompensa_velocidad,
        "recompensa_verticalidad": recompensa_verticalidad,
        "recompensa_supervivencia": recompensa_supervivencia,
        "penalizacion_torque": penalizacion_torque,
        "penalizacion_potencia": penalizacion_potencia,
        "penalizacion_variacion_accion": penalizacion_variacion,
        "penalizacion_deslizamiento_pies": penalizacion_deslizamiento,
        "penalizacion_colision_cuerpo": penalizacion_colision,
        "recompensa_total": recompensa_total,
    }
    return recompensa_total, componentes


def calcular_terminacion(
    estado: EstadoRobot,
    configuracion: ConfiguracionCaminarBasico,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Determina si un entorno debe terminar.

    Criterios básicos:
    - base demasiado baja
    - robot demasiado inclinado
    """

    base_demasiado_baja = estado.altura_base < configuracion.altura_minima_base

    coseno_vertical = torch.abs(estado.gravedad_proyectada[:, 2])
    inclinacion_excesiva = coseno_vertical < configuracion.coseno_minimo_vertical

    termino = base_demasiado_baja | inclinacion_excesiva

    detalles = {
        "base_demasiado_baja": base_demasiado_baja,
        "inclinacion_excesiva": inclinacion_excesiva,
        "termino": termino,
    }
    return termino, detalles


def aplicar_penalizacion_por_caida(
    recompensa: Tensor,
    termino: Tensor,
    configuracion: ConfiguracionCaminarBasico,
) -> Tensor:
    """Resta una penalización extra a los entornos que terminaron por caída."""

    penalizacion = torch.where(
        termino,
        torch.full_like(recompensa, configuracion.penalizacion_caida),
        torch.zeros_like(recompensa),
    )
    return recompensa + penalizacion
