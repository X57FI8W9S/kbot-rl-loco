#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def validar_event_files(run_dir: Path) -> None:
    event_files = sorted(
        run_dir.glob("events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not event_files:
        raise FileNotFoundError(f"No se encontro ningun events.out.tfevents* en: {run_dir}")


def cargar_series_episode(run_dir: Path) -> tuple[list[int], dict[str, list[float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("No se pudo importar tensorboard. Ejecuta este script desde la .venv del repo.") from exc

    acumulador = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acumulador.Reload()

    tags = [
        tag
        for tag in acumulador.Tags().get("scalars", [])
        if tag.startswith("Episode/")
    ]
    if not tags:
        raise RuntimeError(f"El run no contiene scalars Episode/: {run_dir}")

    pasos_base: list[int] | None = None
    series: dict[str, list[float]] = {}
    for tag in tags:
        eventos = acumulador.Scalars(tag)
        pasos = [evento.step for evento in eventos]
        valores = [float(evento.value) for evento in eventos]

        if pasos_base is None:
            pasos_base = pasos
        elif pasos != pasos_base:
            raise RuntimeError(f"Los pasos de {tag} no coinciden con la serie base")

        series[tag.removeprefix("Episode/")] = valores

    if pasos_base is None:
        raise RuntimeError(f"No se pudieron leer series desde: {run_dir}")
    return pasos_base, series


def escribir_csv(pasos: list[int], series: dict[str, list[float]], ruta_csv: Path) -> None:
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    nombres_columnas = ["iteracion", *series.keys()]

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=nombres_columnas)
        writer.writeheader()
        for indice, paso in enumerate(pasos):
            fila = {"iteracion": paso}
            for nombre, valores in series.items():
                fila[nombre] = valores[indice]
            writer.writerow(fila)


def convertir_run_a_csv(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"No existe el directorio del run: {run_dir}")

    validar_event_files(run_dir)
    pasos, series = cargar_series_episode(run_dir)
    ruta_csv = run_dir / "metricas" / "recompensas_componentes.csv"
    escribir_csv(pasos, series, ruta_csv)
    return ruta_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte scalars Episode/ de TensorBoard a CSV.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Directorio del run de entrenamiento.")
    args = parser.parse_args()

    ruta_csv = convertir_run_a_csv(args.run_dir)
    print(f"[INFO] CSV de recompensas escrito en: {ruta_csv}", flush=True)


if __name__ == "__main__":
    main()
