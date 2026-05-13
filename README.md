# TSP Genetic Algorithm Simulator

A desktop GUI application for solving the **Travelling Salesman Problem (TSP)** using a configurable **Genetic Algorithm (GA)**. Built with Python and Tkinter.

## Features

- **TSP Instance Loading** — Select from bundled benchmark instances (`berlin52`, `eil51`, `kroA100`, `ch130`) with known optimal costs
- **Configurable GA Parameters**:
  - Population size and number of generations
  - Elitist strategy toggle
  - Optional Hill Climbing (2-opt / 3-opt) with configurable start generation
- **Selection Operators**: Tournament, Custom Roulette
- **Crossover Operators**: OX (Order), CX (Cycle), PMX (Partially Mapped), Custom
- **Mutation Operators**: 2-swap, Shift, Scramble, Inversion, Custom
- **Visualization**:
  - Real-time instance map with city nodes and tour edges
  - Convergence chart tracking best and average tour length per generation
- **Experiment Support** — Run single or batch experiments with optional seed control

## Requirements

- Python 3.10+
- Tkinter (included with most Python distributions)

## Installation

```bash
git clone https://github.com/Ar1Mi/TSP-SYM.git
cd TSP-SYM
```

### macOS (Homebrew Python)

If you encounter `ModuleNotFoundError: No module named '_tkinter'`, install the Tkinter package:

```bash
brew install python-tk@3.14
```

## Usage

```bash
python3 app.py
```

The simulator window will open with a control panel on the left and visualization area on the right.

## Project Structure

```
.
├── app.py          # Main application — GUI and simulator logic
└── README.md       # Project documentation
```

## Architecture

The application follows a single-class MVC-like architecture:

- **`TSPSimulatorUI`** — Manages the entire GUI layout, state variables, and event wiring
- **Controls Panel** — Left sidebar with parameter inputs organized into logical groups
- **Visual Panel** — Right area with the instance map canvas and convergence chart
- **Footer** — Status bar and credits

## Roadmap

- [ ] Parse `.tsp` files and render real city coordinates
- [ ] Implement GA engine (selection, crossover, mutation, evaluation)
- [ ] Connect algorithm output to the convergence chart
- [ ] Display best tour on the instance map in real time
- [ ] Export experiment results to CSV

## Authors

- **Artur Gubanovich**
- **Yahor Falkouski**

## License

This project is developed as part of the MIO (Metaheuristics and Intelligent Optimization) course.
