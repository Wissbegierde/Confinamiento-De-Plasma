# Confinamiento de plasma — simulador PIC

Proyecto de simulación de confinamiento de plasma con **motor PIC** (Poisson + Boris), **motor lite** vectorizado, análisis **Monte Carlo**, **mapas de calor** alineados con la misma corrida PIC, e **interfaz gráfica** (Tkinter).

## Requisitos

- **Python** 3.10+ (recomendado 3.11–3.13)
- Paquetes:

```bash
pip install numpy scipy matplotlib joblib
```

- **Tkinter**: suele venir con Python en Windows; en Linux: `sudo apt install python3-tk`.

## Inicio rápido

```bash
cd Lib
python main.py
```

El menú interactivo permite elegir motor (PIC o lite), geometría, campos, especies, pasos y si guardar trayectorias, figuras Monte Carlo, mapas de calor y la ventana 3D.

Otras opciones:

```bash
python main.py --list          # últimas corridas guardadas
python main.py --list 30
python main.py --figuras "..\data\simulaciones\<run_id>"   # regenerar PNG MC
```

## Interfaz gráfica (GUI)

Ventana completa con parámetros, vista 3D embebida, curvas de decaimiento y energía:

```bash
cd Lib
python gui_principal.py
```

**Vista 3D:** rueda sobre el gráfico para zoom; **clic izquierdo** (o central) + arrastre para **rotar** (incluye vistas desde arriba y abajo); **clic derecho** + arrastre para **pan**. Requiere backend **TkAgg** (Matplotlib).

Si la 3D no responde, asegúrate de que el cursor está encima del panel 3D al usar la rueda o al arrastrar.

## Visualización 3D (solo Matplotlib)

Al terminar una corrida con `main.py`, si activas **visualización 3D**, se abre `visualizacion.py` (`lanzar_visualizacion`): slider, play/pausa y flechas de **E** / **B** cuando corresponda.

## Estructura del proyecto

| Ruta | Descripción |
|------|-------------|
| `Lib/main.py` | Entrada principal, carpetas `data/simulaciones/<run_id>/` |
| `Lib/gui_principal.py` | Interfaz gráfica Tkinter |
| `Lib/motor.py` | PIC con Poisson |
| `Lib/motor_lite.py` | Integración vectorizada sin rejilla |
| `Lib/montecarlo.py` | τ, decaimiento, gráficas, censura exponencial |
| `Lib/mapas_calor.py` | Densidad e impactos desde la corrida PIC |
| `Lib/visualizacion.py` | Animación 3D Matplotlib |
| `Lib/contenedor.py`, `Lib/campos.py` | Geometrías y campos externos |
| `data/simulaciones/configuraciones_recomendadas.txt` | Presets por geometría (subido al repo) |

Las **corridas completas** (CSV, PNG, caché) se ignoran en Git por `.gitignore`; solo se versiona la guía de configuraciones anterior.

## Tests y verificación

Desde la carpeta `Lib`:

```bash
python test_montecarlo.py    # suite principal (montecarlo + motor lite + boris)
python test_colisiones.py    # Maxwell-Boltzmann y colisiones estocásticas
python verificar_corrida.py  # corrida corta y comprobaciones de coherencia
python validacion_fisica.py  # Larmor, ciclotrón, Bohm, deriva de energía → data/validacion/
```

Utilidades compartidas de los tests: `Lib/test_helpers.py` (sin duplicar colores ni el resumen final).

Diagnóstico de escalas (`dt` vs período ciclotrón):

```bash
python _diagnostico_escalas.py
```

## Documentación de parámetros

Ver **`data/simulaciones/configuraciones_recomendadas.txt`**: valores sugeridos por geometría (cilindro, esfera, caja, placas, tokamak), unidades de tiempo para las gráficas y cómo interpretar τ censurado vs histograma de escapes.

## Licencia y autoría

Repositorio: [Confinamiento-De-Plasma](https://github.com/Wissbegierde/Confinamiento-De-Plasma) (rama `main`). Ajusta aquí licencia y créditos del curso/proyecto si aplica.
