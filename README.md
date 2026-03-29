# Robotica KBot Isaac

Repositorio limpio para trabajo con KBot sobre Isaac Lab.

## Objetivo

Separar:

- instalacion base de Isaac Lab
- codigo propio del proyecto
- assets redistribuibles
- pruebas y salidas


## Estructura

```text
robotica_kbot_isaac/
├── setup/
├── isaac_lab/
├── codigo/
├── scripts/
├── assets/
├── pruebas/
├── salidas/
└── docs/
```

## Siguientes pasos

1. Instalar Isaac Lab limpio dentro de `isaac_lab/IsaacLab`.
2. Activar el entorno virtual desde `setup/activar.sh`.
3. Migrar los archivos de prueba desde `kbot-rl-loco` hacia `codigo/` y `scripts/`.
4. Recién después evaluar si conviene renombrar `kbot-rl-loco` a `kbot-rl-loco(old)`.
