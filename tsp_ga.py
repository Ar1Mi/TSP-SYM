from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

City = tuple[float, float]
Route = list[int]


@dataclass(frozen=True)
class BinaryEncodingConfig:
    min_value: float
    max_value: float
    bits: int


@dataclass(frozen=True)
class TournamentRound:
    candidate_indices: tuple[int, ...]
    candidate_costs: tuple[float, ...]
    winner_index: int
    winner_cost: float


@dataclass(frozen=True)
class ParentSelectionResult:
    draws: tuple[float, ...]
    selected_indices: tuple[int, ...]
    dropped_index: int | None


@dataclass(frozen=True)
class CrossoverDebug:
    cut_left: int
    cut_right: int
    copied_segment: tuple[int, ...]
    fill_order_from_parent_b: tuple[int, ...]


@dataclass(frozen=True)
class MutationDebug:
    swap_i: int
    swap_j: int
    before: tuple[int, ...]
    after: tuple[int, ...]


def _validate_binary_config(config: BinaryEncodingConfig) -> None:
    if config.bits < 1:
        raise ValueError("bits must be >= 1")
    if config.max_value <= config.min_value:
        raise ValueError("max_value must be greater than min_value")


def encode_real_to_binary(value: float, config: BinaryEncodingConfig) -> str:
    _validate_binary_config(config)
    if value < config.min_value or value > config.max_value:
        raise ValueError("value is outside configured range")

    levels = (1 << config.bits) - 1
    normalized = (value - config.min_value) / (config.max_value - config.min_value)
    encoded_int = round(normalized * levels)
    return format(encoded_int, f"0{config.bits}b")


def decode_binary_to_real(binary: str, config: BinaryEncodingConfig) -> float:
    _validate_binary_config(config)
    if len(binary) != config.bits:
        raise ValueError("binary length does not match config.bits")
    if any(ch not in "01" for ch in binary):
        raise ValueError("binary must contain only '0' and '1'")

    levels = (1 << config.bits) - 1
    encoded_int = int(binary, 2)
    normalized = encoded_int / levels
    return config.min_value + normalized * (config.max_value - config.min_value)


def is_valid_permutation(route: Sequence[int], num_cities: int) -> bool:
    return len(route) == num_cities and set(route) == set(range(num_cities))


def build_distance_matrix(cities: Sequence[City]) -> list[list[float]]:
    n = len(cities)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        xi, yi = cities[i]
        for j in range(i + 1, n):
            xj, yj = cities[j]
            dist = math.hypot(xi - xj, yi - yj)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def route_length(route: Sequence[int], distance_matrix: Sequence[Sequence[float]]) -> float:
    if not route:
        raise ValueError("route must not be empty")
    total = 0.0
    for idx in range(len(route)):
        a = route[idx]
        b = route[(idx + 1) % len(route)]
        total += distance_matrix[a][b]
    return total


def generate_population(pop_size: int, num_cities: int, rng: random.Random) -> list[Route]:
    if pop_size < 1:
        raise ValueError("pop_size must be >= 1")
    if num_cities < 2:
        raise ValueError("num_cities must be >= 2")

    base = list(range(num_cities))
    population: list[Route] = []
    for _ in range(pop_size):
        route = base.copy()
        rng.shuffle(route)
        population.append(route)
    return population


def tournament_selection_indices(
    costs: Sequence[float],
    tournament_size: int,
    rng: random.Random,
    num_parents: int = 1,
) -> tuple[list[int], list[TournamentRound]]:
    if not costs:
        raise ValueError("costs must not be empty")
    if num_parents < 1:
        raise ValueError("num_parents must be >= 1")

    population_size = len(costs)
    draw_size = max(1, min(tournament_size, population_size))
    all_indices = list(range(population_size))

    selected: list[int] = []
    rounds: list[TournamentRound] = []
    for _ in range(num_parents):
        sampled = rng.sample(all_indices, k=draw_size)
        winner = min(sampled, key=lambda idx: costs[idx])
        round_debug = TournamentRound(
            candidate_indices=tuple(sampled),
            candidate_costs=tuple(costs[idx] for idx in sampled),
            winner_index=winner,
            winner_cost=costs[winner],
        )
        selected.append(winner)
        rounds.append(round_debug)
    return selected, rounds


def roulette_selection_indices(
    costs: Sequence[float], rng: random.Random, num_parents: int = 1
) -> list[int]:
    if not costs:
        raise ValueError("costs must not be empty")
    if num_parents < 1:
        raise ValueError("num_parents must be >= 1")

    epsilon = 1e-12
    fitness = [1.0 / (cost + epsilon) for cost in costs]
    total_fitness = sum(fitness)

    selected: list[int] = []
    for _ in range(num_parents):
        draw = rng.random() * total_fitness
        running = 0.0
        winner = len(costs) - 1
        for idx, fit in enumerate(fitness):
            running += fit
            if running >= draw:
                winner = idx
                break
        selected.append(winner)
    return selected


def select_parents_by_crossover_probability(
    population_size: int, p_cross: float, rng: random.Random
) -> ParentSelectionResult:
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if p_cross < 0.0 or p_cross > 1.0:
        raise ValueError("p_cross must be in [0, 1]")

    draws = tuple(rng.random() for _ in range(population_size))
    selected = [idx for idx, draw in enumerate(draws) if draw < p_cross]

    dropped: int | None = None
    if len(selected) % 2 == 1:
        dropped = selected.pop()

    return ParentSelectionResult(
        draws=draws,
        selected_indices=tuple(selected),
        dropped_index=dropped,
    )


def ordered_crossover(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    rng: random.Random,
    cut_left: int | None = None,
    cut_right: int | None = None,
) -> tuple[Route, CrossoverDebug]:
    if len(parent_a) != len(parent_b):
        raise ValueError("both parents must have equal length")
    n = len(parent_a)
    if n < 2:
        raise ValueError("parents must contain at least 2 genes")

    if cut_left is None or cut_right is None:
        cut_left, cut_right = sorted(rng.sample(range(n), k=2))
    if cut_left > cut_right:
        cut_left, cut_right = cut_right, cut_left
    if cut_left < 0 or cut_right >= n or cut_left == cut_right:
        raise ValueError("invalid crossover cuts")

    child: list[int | None] = [None] * n
    segment = list(parent_a[cut_left : cut_right + 1])
    child[cut_left : cut_right + 1] = segment

    fill_values = [gene for gene in parent_b if gene not in segment]
    fill_positions = list(range(cut_right + 1, n)) + list(range(0, cut_left))
    for pos, gene in zip(fill_positions, fill_values):
        child[pos] = gene

    if any(gene is None for gene in child):
        raise RuntimeError("ordered crossover failed to fill child")

    child_route: Route = [int(gene) for gene in child]
    debug = CrossoverDebug(
        cut_left=cut_left,
        cut_right=cut_right,
        copied_segment=tuple(segment),
        fill_order_from_parent_b=tuple(fill_values),
    )
    return child_route, debug


def mutate_swap(
    route: Sequence[int], rng: random.Random, swap_i: int | None = None, swap_j: int | None = None
) -> tuple[Route, MutationDebug]:
    n = len(route)
    if n < 2:
        raise ValueError("route must contain at least 2 cities")

    if swap_i is None or swap_j is None:
        swap_i, swap_j = rng.sample(range(n), k=2)
    if swap_i == swap_j:
        raise ValueError("swap_i and swap_j must be different")
    if not (0 <= swap_i < n and 0 <= swap_j < n):
        raise ValueError("swap indices are out of bounds")

    before = list(route)
    after = list(route)
    after[swap_i], after[swap_j] = after[swap_j], after[swap_i]

    debug = MutationDebug(
        swap_i=swap_i,
        swap_j=swap_j,
        before=tuple(before),
        after=tuple(after),
    )
    return after, debug

