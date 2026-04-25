#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
RUTA_CODIGO = Path(__file__).resolve().parents[1]
if str(RUTA_CODIGO) not in sys.path:
    sys.path.insert(0, str(RUTA_CODIGO))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RECOMPENSAS_INTERMEDIAS = [
    "aporte_vx",
    "aporte_x",
    "aporte_vy",
    "aporte_yaw",
    "aporte_vert",
    "aporte_superv",
]
PENALIZACIONES_INTERMEDIAS = [
    "costo_y",
    "costo_suavidad",
    "costo_torque",
    "costo_pose",
]
RECOMPENSAS_TERMINALES = [
    "bonus_final_superv",
    "bonus_final_x",
]
PENALIZACIONES_TERMINALES = [
    "malus_final_vx",
    "malus_final_caida",
]

ETIQUETAS = {
    "aporte_vx": "aporte_vx",
    "aporte_x": "aporte_x",
    "aporte_vy": "aporte_vy",
    "aporte_yaw": "aporte_yaw",
    "aporte_vert": "aporte_vert",
    "aporte_superv": "aporte_superv",
    "costo_y": "costo_y",
    "costo_suavidad": "costo_suavidad",
    "costo_torque": "costo_torque",
    "costo_pose": "costo_pose",
    "bonus_final_superv": "bonus_final_superv",
    "bonus_final_x": "bonus_final_x",
    "malus_final_vx": "malus_final_vx",
    "malus_final_caida": "malus_final_caida",
    "total": "total",
}


def cargar_csv(ruta_csv: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de recompensas: {ruta_csv}")

    with ruta_csv.open(newline="", encoding="utf-8") as archivo:
        reader = csv.DictReader(archivo)
        if reader.fieldnames is None or "iteracion" not in reader.fieldnames:
            raise RuntimeError(f"El CSV debe incluir una columna 'iteracion': {ruta_csv}")

        columnas = {nombre: [] for nombre in reader.fieldnames}
        for fila in reader:
            for nombre in columnas:
                columnas[nombre].append(float(fila[nombre]))

    iteraciones = np.array(columnas.pop("iteracion"), dtype=float)
    series = {nombre: np.array(valores, dtype=float) for nombre, valores in columnas.items()}
    return iteraciones, series


def validar_columnas(series: dict[str, np.ndarray], nombres: list[str]) -> None:
    faltantes = [nombre for nombre in nombres if nombre not in series]
    if faltantes:
        raise RuntimeError("Faltan columnas en recompensas_componentes.csv: " + ", ".join(faltantes))


def ordenar_por_contribucion(
    nombres: list[str],
    series: dict[str, np.ndarray],
) -> list[str]:
    return sorted(
        nombres,
        key=lambda nombre: float(np.nanmean(np.abs(series[nombre]))),
    )


def graficar_contribuciones(
    iteraciones: np.ndarray,
    series: dict[str, np.ndarray],
    recompensas: list[str],
    penalizaciones: list[str],
    titulo: str,
    ruta_salida: Path,
    colores_penalizaciones: str = "GnBu",
) -> None:
    validar_columnas(series, recompensas + penalizaciones)
    recompensas = ordenar_por_contribucion(recompensas, series)
    penalizaciones = ordenar_por_contribucion(penalizaciones, series)

    valores_positivos = [series[nombre] for nombre in recompensas]
    valores_negativos = [series[nombre] for nombre in penalizaciones]
    acumulado_positivo = np.cumsum(np.vstack(valores_positivos), axis=0)
    acumulado_negativo = -np.cumsum(np.vstack(valores_negativos), axis=0)
    total = acumulado_positivo[-1] + acumulado_negativo[-1]

    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)

    colores_recompensas = plt.cm.Greens(np.linspace(0.3, 0.95, len(recompensas)))
    base = np.zeros_like(iteraciones, dtype=float)
    for nombre, techo, color in zip(recompensas, acumulado_positivo, colores_recompensas):
        ax.fill_between(iteraciones, base, techo, label=ETIQUETAS[nombre], color=color, alpha=0.9)
        base = techo

    cmap_penalizaciones = getattr(plt.cm, colores_penalizaciones)
    colores = cmap_penalizaciones(np.linspace(0.3, 0.95, len(penalizaciones)))
    techo = np.zeros_like(iteraciones, dtype=float)
    for nombre, base, color in zip(penalizaciones, acumulado_negativo, colores):
        ax.fill_between(iteraciones, techo, base, label=ETIQUETAS[nombre], color=color, alpha=0.9)
        techo = base

    ax.plot(iteraciones, total, color="black", linewidth=2.0, label=ETIQUETAS["total"])
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Recompensa")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generar_graficos(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    ruta_csv = run_dir / "metricas" / "recompensas_componentes.csv"
    iteraciones, series = cargar_csv(ruta_csv)
    graficos_dir = run_dir / "graficos"

    salidas = [
        graficos_dir / "recompensas_intermedias.png",
        graficos_dir / "recompensas_terminales.png",
        graficos_dir / "recompensas_combinadas.png",
    ]
    graficar_contribuciones(
        iteraciones,
        series,
        RECOMPENSAS_INTERMEDIAS,
        PENALIZACIONES_INTERMEDIAS,
        "Recompensas intermedias",
        salidas[0],
    )
    graficar_contribuciones(
        iteraciones,
        series,
        RECOMPENSAS_TERMINALES,
        PENALIZACIONES_TERMINALES,
        "Recompensas terminales",
        salidas[1],
    )
    graficar_contribuciones(
        iteraciones,
        series,
        RECOMPENSAS_INTERMEDIAS + RECOMPENSAS_TERMINALES,
        PENALIZACIONES_INTERMEDIAS + PENALIZACIONES_TERMINALES,
        "Componentes recompensa desagregada",
        salidas[2],
        colores_penalizaciones="GnBu",
    )
    return salidas


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera graficos de recompensas desde recompensas_componentes.csv.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directorio del run de entrenamiento.")
    args = parser.parse_args()

    salidas = generar_graficos(args.run_dir)
    for ruta in salidas:
        print(f"[INFO] Grafico escrito en: {ruta}", flush=True)


if __name__ == "__main__":
    main()
