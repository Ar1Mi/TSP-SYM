from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from statistics import mean
from typing import Callable, Sequence

from tsp_ga import (
    Route,
    build_distance_matrix,
    generate_population,
    ordered_crossover,
    route_length,
    roulette_selection_indices,
    tournament_selection_indices,
)

City = tuple[float, float]


@dataclass(frozen=True)
class TSPInstance:
    name: str
    cities: tuple[City, ...]
    optimal_cost: float | None = None
    reference_route: tuple[int, ...] | None = None
    source: str = "synthetic"

    @property
    def dimension(self) -> int:
        return len(self.cities)


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    num_generations: int
    selection_method: str = "tournament"
    tournament_size: int = 3
    crossover_method: str = "ox"
    crossover_prob: float = 0.80
    mutation_method: str = "2-swap"
    mutation_prob: float = 0.01
    elitist: bool = True
    hillclimbing: bool = False
    hillclimbing_variant: str = "2-opt"
    hillclimbing_start_generation: int = 25
    seed: int | None = None


@dataclass(frozen=True)
class GenerationSnapshot:
    generation: int
    best_cost: float
    avg_cost: float
    best_route: tuple[int, ...]


@dataclass(frozen=True)
class GARunResult:
    instance: TSPInstance
    config: GAConfig
    best_route: tuple[int, ...]
    best_cost: float
    best_generation: int
    history_best: tuple[float, ...]
    history_avg: tuple[float, ...]
    snapshots: tuple[GenerationSnapshot, ...]


_BUNDLED_SPECS: dict[str, tuple[int, int, float | None]] = {
    "berlin52.tsp": (52, 52, 7542.0),
    "eil51.tsp": (51, 51, 426.0),
    "kroA100.tsp": (100, 100, 21282.0),
    "ch130.tsp": (130, 130, 6110.0),
}


def bundled_instance_names() -> list[str]:
    return list(_BUNDLED_SPECS.keys())


def load_instance_by_name(name: str, project_root: Path | None = None) -> TSPInstance:
    """
    Resolve instance by:
      1) explicit path if it exists
      2) ./instances/<name> if file exists
      3) built-in synthetic fallback with matching dimension
    """
    direct_path = Path(name)
    if direct_path.exists():
        return load_tsplib_instance(direct_path)

    if project_root is not None:
        candidate = project_root / "instances" / name
        if candidate.exists():
            return load_tsplib_instance(candidate)

    if name not in _BUNDLED_SPECS:
        available = ", ".join(bundled_instance_names())
        raise ValueError(f"Unknown instance '{name}'. Available: {available}")

    dimension, seed, optimal = _BUNDLED_SPECS[name]
    cities = _synthetic_cities(dimension=dimension, seed=seed)
    reference_route = _build_reference_route(cities)
    return TSPInstance(
        name=name,
        cities=tuple(cities),
        optimal_cost=optimal,
        reference_route=tuple(reference_route),
        source="synthetic",
    )


def load_tsplib_instance(path: str | Path) -> TSPInstance:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    name = file_path.name
    dimension: int | None = None
    edge_weight_type = "EUC_2D"
    coords: list[City] = []
    in_coord_section = False

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()

            if upper.startswith("EOF"):
                break
            if upper.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            if not in_coord_section:
                if ":" in line:
                    key, value = [part.strip() for part in line.split(":", 1)]
                else:
                    parts = line.split(maxsplit=1)
                    key = parts[0]
                    value = parts[1] if len(parts) > 1 else ""
                key_upper = key.upper()
                if key_upper == "NAME" and value:
                    name = value
                elif key_upper == "DIMENSION":
                    dimension = int(value)
                elif key_upper == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value.upper()
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            # Format: <index> <x> <y>
            x = float(parts[1])
            y = float(parts[2])
            coords.append((x, y))

    if edge_weight_type not in {"EUC_2D", "CEIL_2D"}:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")
    if dimension is None:
        dimension = len(coords)
    if dimension != len(coords):
        raise ValueError(
            f"Invalid TSPLIB file: DIMENSION={dimension}, NODE_COORD_SECTION rows={len(coords)}"
        )
    if len(coords) < 2:
        raise ValueError("TSPLIB instance must contain at least 2 cities")

    optimal = _BUNDLED_SPECS.get(file_path.name, (0, 0, None))[2]
    reference_route = _build_reference_route(coords)
    return TSPInstance(
        name=name,
        cities=tuple(coords),
        optimal_cost=optimal,
        reference_route=tuple(reference_route),
        source=str(file_path),
    )


def _synthetic_cities(dimension: int, seed: int) -> list[City]:
    rng = random.Random(seed)
    cities: list[City] = []
    center_x = 900.0
    center_y = 600.0

    for idx in range(dimension):
        angle = (2.0 * math.pi * idx / dimension) + rng.uniform(-0.18, 0.18)
        radius = 380.0 + 90.0 * math.sin(idx * 0.31) + rng.uniform(-70.0, 70.0)
        x = center_x + radius * math.cos(angle) + rng.uniform(-50.0, 50.0)
        y = center_y + radius * math.sin(angle) + rng.uniform(-50.0, 50.0)
        cities.append((x, y))
    return cities


def _build_reference_route(cities: Sequence[City]) -> list[int]:
    if not cities:
        return []
    cx = mean(point[0] for point in cities)
    cy = mean(point[1] for point in cities)
    return sorted(
        range(len(cities)),
        key=lambda idx: math.atan2(cities[idx][1] - cy, cities[idx][0] - cx),
    )


def run_ga(
    instance: TSPInstance,
    config: GAConfig,
    progress_callback: Callable[[GenerationSnapshot], None] | None = None,
) -> GARunResult:
    _validate_ga_config(instance, config)
    rng = random.Random(config.seed)
    distance_matrix = build_distance_matrix(instance.cities)

    population = generate_population(config.population_size, instance.dimension, rng)
    costs = [route_length(route, distance_matrix) for route in population]

    history_best: list[float] = []
    history_avg: list[float] = []
    snapshots: list[GenerationSnapshot] = []

    best_index = min(range(len(population)), key=lambda idx: costs[idx])
    best_route = population[best_index].copy()
    best_cost = costs[best_index]
    best_generation = 0

    initial_snapshot = GenerationSnapshot(
        generation=0,
        best_cost=best_cost,
        avg_cost=mean(costs),
        best_route=tuple(best_route),
    )
    history_best.append(initial_snapshot.best_cost)
    history_avg.append(initial_snapshot.avg_cost)
    snapshots.append(initial_snapshot)
    if progress_callback is not None:
        progress_callback(initial_snapshot)

    for generation in range(1, config.num_generations + 1):
        population = _create_next_generation(
            population=population,
            costs=costs,
            distance_matrix=distance_matrix,
            config=config,
            rng=rng,
            generation=generation,
        )
        costs = [route_length(route, distance_matrix) for route in population]

        best_idx = min(range(len(population)), key=lambda idx: costs[idx])
        gen_best_cost = costs[best_idx]
        gen_avg_cost = mean(costs)
        gen_best_route = population[best_idx].copy()

        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_route = gen_best_route.copy()
            best_generation = generation

        snapshot = GenerationSnapshot(
            generation=generation,
            best_cost=gen_best_cost,
            avg_cost=gen_avg_cost,
            best_route=tuple(gen_best_route),
        )
        history_best.append(gen_best_cost)
        history_avg.append(gen_avg_cost)
        snapshots.append(snapshot)
        if progress_callback is not None:
            progress_callback(snapshot)

    return GARunResult(
        instance=instance,
        config=config,
        best_route=tuple(best_route),
        best_cost=best_cost,
        best_generation=best_generation,
        history_best=tuple(history_best),
        history_avg=tuple(history_avg),
        snapshots=tuple(snapshots),
    )


def _validate_ga_config(instance: TSPInstance, config: GAConfig) -> None:
    if config.population_size < 2:
        raise ValueError("Population size must be >= 2")
    if config.num_generations < 1:
        raise ValueError("Number of generations must be >= 1")
    if config.tournament_size < 1:
        raise ValueError("Tournament size must be >= 1")
    if not (0.0 <= config.crossover_prob <= 1.0):
        raise ValueError("Crossover probability must be in [0, 1]")
    if not (0.0 <= config.mutation_prob <= 1.0):
        raise ValueError("Mutation probability must be in [0, 1]")
    if config.hillclimbing_start_generation < 0:
        raise ValueError("Generation to start hill climbing must be >= 0")
    if instance.dimension < 2:
        raise ValueError("Instance dimension must be >= 2")


def _create_next_generation(
    population: Sequence[Route],
    costs: Sequence[float],
    distance_matrix: Sequence[Sequence[float]],
    config: GAConfig,
    rng: random.Random,
    generation: int,
) -> list[Route]:
    pop_size = len(population)
    new_population: list[Route] = []

    if config.elitist:
        elite_idx = min(range(pop_size), key=lambda idx: costs[idx])
        new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        parent_a = _select_parent(population, costs, config, rng)
        parent_b = _select_parent(population, costs, config, rng)

        child_a = parent_a.copy()
        child_b = parent_b.copy()
        if rng.random() < config.crossover_prob:
            child_a, child_b = _crossover_pair(parent_a, parent_b, config.crossover_method, rng)

        if rng.random() < config.mutation_prob:
            child_a = _mutate_route(child_a, config.mutation_method, rng)
        if rng.random() < config.mutation_prob:
            child_b = _mutate_route(child_b, config.mutation_method, rng)

        if config.hillclimbing and generation >= config.hillclimbing_start_generation:
            child_a = _hill_climb(child_a, distance_matrix, config.hillclimbing_variant)
            child_b = _hill_climb(child_b, distance_matrix, config.hillclimbing_variant)

        new_population.append(child_a)
        if len(new_population) < pop_size:
            new_population.append(child_b)

    return new_population


def _select_parent(
    population: Sequence[Route], costs: Sequence[float], config: GAConfig, rng: random.Random
) -> Route:
    method = config.selection_method.lower()
    if method == "tournament":
        selected, _ = tournament_selection_indices(
            costs=costs, tournament_size=config.tournament_size, rng=rng, num_parents=1
        )
        return population[selected[0]]
    if method in {"my-roulette", "roulette"}:
        selected = roulette_selection_indices(costs=costs, rng=rng, num_parents=1)
        return population[selected[0]]
    # Fallback to tournament if unknown method was selected in UI.
    selected, _ = tournament_selection_indices(
        costs=costs, tournament_size=config.tournament_size, rng=rng, num_parents=1
    )
    return population[selected[0]]


def _crossover_pair(
    parent_a: Sequence[int], parent_b: Sequence[int], method: str, rng: random.Random
) -> tuple[Route, Route]:
    method_norm = method.lower()
    if method_norm in {"ox", "my-crossover"}:
        child_a, debug = ordered_crossover(parent_a, parent_b, rng=rng)
        child_b, _ = ordered_crossover(
            parent_b, parent_a, rng=rng, cut_left=debug.cut_left, cut_right=debug.cut_right
        )
        return child_a, child_b
    if method_norm == "cx":
        return _cycle_crossover_pair(parent_a, parent_b)
    if method_norm == "pmx":
        return _pmx_crossover_pair(parent_a, parent_b, rng)
    # Fallback
    child_a, debug = ordered_crossover(parent_a, parent_b, rng=rng)
    child_b, _ = ordered_crossover(
        parent_b, parent_a, rng=rng, cut_left=debug.cut_left, cut_right=debug.cut_right
    )
    return child_a, child_b


def _cycle_crossover_pair(parent_a: Sequence[int], parent_b: Sequence[int]) -> tuple[Route, Route]:
    n = len(parent_a)
    child_a: list[int | None] = [None] * n
    child_b: list[int | None] = [None] * n
    visited = [False] * n
    cycle_index = 0

    index_by_value_a = {value: idx for idx, value in enumerate(parent_a)}

    while not all(visited):
        start = visited.index(False)
        cycle_positions: list[int] = []
        idx = start
        while not visited[idx]:
            visited[idx] = True
            cycle_positions.append(idx)
            idx = index_by_value_a[parent_b[idx]]

        use_parent_a = cycle_index % 2 == 0
        for pos in cycle_positions:
            if use_parent_a:
                child_a[pos] = parent_a[pos]
                child_b[pos] = parent_b[pos]
            else:
                child_a[pos] = parent_b[pos]
                child_b[pos] = parent_a[pos]
        cycle_index += 1

    return [int(g) for g in child_a], [int(g) for g in child_b]


def _pmx_crossover_pair(
    parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random
) -> tuple[Route, Route]:
    n = len(parent_a)
    left, right = sorted(rng.sample(range(n), k=2))
    child_a = _pmx_child(parent_a, parent_b, left, right)
    child_b = _pmx_child(parent_b, parent_a, left, right)
    return child_a, child_b


def _pmx_child(parent_segment_source: Sequence[int], parent_fill_source: Sequence[int], left: int, right: int) -> Route:
    n = len(parent_segment_source)
    child: list[int | None] = [None] * n

    segment_a = list(parent_segment_source[left : right + 1])
    child[left : right + 1] = segment_a

    segment_values = set(segment_a)
    index_in_parent_segment = {value: idx for idx, value in enumerate(parent_segment_source)}

    for idx in list(range(0, left)) + list(range(right + 1, n)):
        gene = parent_fill_source[idx]
        visited: set[int] = set()
        while gene in segment_values:
            if gene in visited:
                break
            visited.add(gene)
            mapped_position = index_in_parent_segment[gene]
            gene = parent_fill_source[mapped_position]

        if gene in segment_values or gene in child:
            for candidate in parent_fill_source:
                if candidate not in child:
                    gene = candidate
                    break
        child[idx] = gene

    return [int(gene) for gene in child]


def _mutate_route(route: Route, mutation_method: str, rng: random.Random) -> Route:
    method = mutation_method.lower()
    n = len(route)
    if n < 2:
        return route.copy()

    if method in {"2-swap", "my-mutation"}:
        return _mutate_swap_only(route, rng)
    if method == "shift":
        i, j = sorted(rng.sample(range(n), k=2))
        gene = route.pop(j)
        route.insert(i, gene)
        return route
    if method == "scramble":
        i, j = sorted(rng.sample(range(n), k=2))
        segment = route[i : j + 1]
        rng.shuffle(segment)
        route[i : j + 1] = segment
        return route
    if method == "inversion":
        i, j = sorted(rng.sample(range(n), k=2))
        route[i : j + 1] = reversed(route[i : j + 1])
        return route
    return _mutate_swap_only(route, rng)


def _mutate_swap_only(route: Route, rng: random.Random) -> Route:
    i, j = rng.sample(range(len(route)), k=2)
    route[i], route[j] = route[j], route[i]
    return route


def _hill_climb(
    route: Route,
    distance_matrix: Sequence[Sequence[float]],
    variant: str,
) -> Route:
    if variant.lower() == "3-opt":
        improved = _two_opt_first_improvement(route, distance_matrix, passes=2)
        return _two_opt_first_improvement(improved, distance_matrix, passes=1)
    return _two_opt_first_improvement(route, distance_matrix, passes=1)


def _two_opt_first_improvement(
    route: Route,
    distance_matrix: Sequence[Sequence[float]],
    passes: int = 1,
) -> Route:
    n = len(route)
    if n < 4:
        return route

    best = route.copy()
    best_cost = route_length(best, distance_matrix)
    for _ in range(max(1, passes)):
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                candidate = best.copy()
                candidate[i + 1 : j + 1] = reversed(candidate[i + 1 : j + 1])
                candidate_cost = route_length(candidate, distance_matrix)
                if candidate_cost + 1e-12 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best
