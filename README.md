# Open RCS

Open RCS is a Python toolkit for Radar Cross Section (RCS) analysis from STL geometry, with monostatic and bistatic solvers.

The project now uses a Jupyter notebook workflow for interactive use. The previous Tkinter desktop widget is no longer part of the active workflow.

## Table of Contents

- [Quick Start (Pixi)](#quick-start-pixi)
- [Use in Jupyter](#use-in-jupyter)
- [Developer Setup](#developer-setup)
- [Help Developers Get Started](#help-developers-get-started)
- [Author Info](#author-info)
- [References](#references)

## Quick Start (Pixi)

1. Install Pixi:
   https://pixi.sh/latest/
2. Clone the repository:

```bash
git clone https://github.com/sanbales/open-rcs
cd open-rcs
```

3. Install environments and dependencies:

```bash
pixi install
```

4. Launch JupyterLab:

```bash
pixi run lab
```

This is the fastest way to start using Open RCS.

## Use in Jupyter

Once JupyterLab starts, open a notebook and use the notebook UI:

```python
from open_rcs import launch_rcs_widget

widget = launch_rcs_widget("..")
```

You can also call the solvers directly:

```python
from open_rcs import (
    AngleSweep,
    MaterialConfig,
    MonostaticSimulationConfig,
    build_geometry_from_stl,
    simulate_monostatic,
)

geometry = build_geometry_from_stl("stl_models/plate.stl", rs_value=0.2)
config = MonostaticSimulationConfig(
    input_model="plate.stl",
    frequency_hz=10.0e9,
    correlation_distance_m=0.0,
    standard_deviation_m=0.0,
    incident_polarization=0,
    angle_sweep=AngleSweep(
        phi_start_deg=0.0,
        phi_stop_deg=90.0,
        phi_step_deg=1.0,
        theta_start_deg=0.0,
        theta_stop_deg=90.0,
        theta_step_deg=1.0,
    ),
    material=MaterialConfig(resistivity_mode=0.2, material_path="materials/example.rcsmat"),
)
result = simulate_monostatic(config, geometry)
```

## Developer Setup

Use the `dev` Pixi environment for lint, type checks, tests, and release tasks:

```bash
pixi run test
```

Coverage reports are generated automatically by pytest and written to:

- `htmlcov/index.html`

## Help Developers Get Started

Common Pixi commands:

| Goal | Command |
|---|---|
| Launch JupyterLab | `pixi run lab` |
| Run tests | `pixi run test` |
| Run lint checks | `pixi run lint` |
| Auto-fix lint issues | `pixi run fix` |
| Format code | `pixi run format` |
| Run type checks | `pixi run typecheck` |
| Run lint + typecheck | `pixi run check` |
| Run full local validation | `pixi run pre-commit` |
| Build package | `pixi run build` |
| Upload package artifacts | `pixi run release` |
| Bump patch version in `pyproject.toml` | `pixi run bump-patch` |
| Install project Jupyter kernel | `pixi run kernel` |

## Software Information

The current version of **Open RCS** consists of a monostatic module implementation, which will serve as the architectural basis for a future bistatic module following planned refactoring and modularization. The simulation employs a **Physical Optics (PO)** approximation that was derived from Jenn's POFACETS. In this method, reflections on each facet are processed as isolated surfaces, excluding multiple reflections and edge diffractions. Each facet is treated as either entirely shadowed or fully illuminated by the incident wave, a binary determination that dictates its contribution to the final RCS calculation. Furthermore, **TMz** and **TEz** polarizations are represented using complex data types in Python 3.

Reflection intensity is calculated via a **Taylor series expansion**, where the regional parameters and the number of terms were initially selected to balance computational requirements. These parameters are guided by external optimization references to ensure accuracy and efficiency within the PO approximation. Using this method, the software generates intensity and electromagnetic power results for each specified direction.

Reflection directions are defined using **spherical coordinates** derived from the input range and sampling of $\phi$ (phi) and $\theta$ (theta). If a single value of $\phi$ is specified for an interval of $\theta$, a projection graph is plotted as a **$\theta$-cut** (and vice versa). In cases where both $\phi$ and $\theta$ are defined over intervals, the software generates a two-dimensional contour map of the **direction cosines**.

## Theoretical Foundation: Radar Cross Section (RCS)

Radar Cross Section (RCS) is a measure of an object's detectability by radar, where a higher value indicates a more visible target. It is determined by several factors, including the ratio of the target's size to the radar wavelength, the material composition (e.g., radar-absorbent materials), incident angles, and the object's geometry. 

Since the invention of radar during World War II, controlling the electromagnetic signature of military platforms—such as aircraft, vessels, and armored vehicles—has been critical to operational success. While the fundamentals of RCS have been understood for decades, calculating the exact electromagnetic signature of complex, real-world targets remains challenging. 

Analytical calculations are only feasible for simple, ideal shapes. Therefore, determining the RCS of complex objects relies on experimental measurements or computational numerical simulations. Numerical simulation is especially valuable when the physical target is unavailable or still in the design phase, allowing for numerous design iterations and material evaluations (RAM application) that would otherwise be impractical.

## Author Info

The software was developed in a Computer and Communications Engineer graduation project as requirement to acquiring a bachelor degree in these areas. The graduating students and advisors involved are mentioned below, with their contact info:

- Amanda Assis Lavinsky (amanda.lavinsky@ime.eb.br)
- 1o Ten Lucas Machado Couto Bezerra (lucas.bezerra@ime.eb.br)
- 1o Ten Mayara Ribeiro Mendonca (mayara.mendonca@ime.eb.br)
- 1o Ten Yu Yi Wang Xia (yu.xia@ime.eb.br)
- 1o Ten Augusto Henrique Goncalves Marques (augusto.henrique12345@protonmail.com)
- 1o Ten Daniel Ambrozio Bretherick Marques (danielbretherick@gmail.com)
- 1o Ten Leticia Vieira da Fonseca (leticiavieiradafonseca@hotmail.com)
- 1o Ten Rafael Pontes tenorio Lima (rafael.pontes882@gmail.com)
- Maj Gabriela Moutinho de Souza Dias (gabriela@ime.eb.br)
- TC Claudio Augusto Barreto Saunders Filho (saunders@ime.eb.br)
- Cel Clayton Escouper das Chagas (escouper@ime.eb.br)

The initial implementation by the CCE team was further extended by Dr. Santiago Balestrini-Robinson (sanbales@gmail.com).

## References

1. Knott, E. F., Shaeffer, J. F., & Tuley, M. T. (1993). *Radar Cross Section*. Artech House.
2. Ruck, G. T., Barrick, D. E., Stuart, W. D., & Krichbaum, C. K. (1970). *Radar Cross Section Handbook* (Vol. 1). Plenum Press/Springer US.
3. Sumithra, P., & Thiripurasundari, D. (2017). A review on computational electromagnetics methods. *Advanced Electromagnetics*, *6*(1), 45–55.
4. Jenn, D. C. (2005). *Radar and Laser Cross Section Engineering* (2nd ed.). American Institute of Aeronautics and Astronautics.
5. Swords, S. S. (1986). *Technical History of the Beginnings of Radar*. Peter Peregrinus Ltd. on behalf of the Institution of Electrical Engineers.
6. Chatzigeorgiadis, F., & Jenn, D. C. (2004). A MATLAB physical-optics RCS prediction code. *IEEE Antennas and Propagation Magazine*, *46*(4), 137–139.

