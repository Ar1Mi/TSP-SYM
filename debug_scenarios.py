from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from tsp_ga import (
    BinaryEncodingConfig,
    encode_real_to_binary,
    decode_binary_to_real,
    generate_population,
    tournament_selection_indices,
    select_parents_by_crossover_probability,
    ordered_crossover,
    mutate_swap,
    is_valid_permutation,
)


@dataclass(frozen=True)
class DebugScenarioResult:
    scenario_id: str
    title: str
    passed: bool
    body: str

    def as_text(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        header = f"[{self.scenario_id}] {self.title} -> {verdict}"
        return f"{header}\n\n{self.body}".strip()


def run_debug_test_1() -> DebugScenarioResult:
    """
    test_1: show correct conversion x(real) <-> x(binary)
    """
    config = BinaryEncodingConfig(min_value=0.0, max_value=100.0, bits=10)
    max_quantization_error = (config.max_value - config.min_value) / ((1 << config.bits) - 1)

    values = [0.0, 12.5, 37.42, 73.0, 99.9]
    lines = [
        "Goal: validate real <-> binary conversion.",
        f"Config: min={config.min_value}, max={config.max_value}, bits={config.bits}.",
        f"Allowed quantization error <= {max_quantization_error:.6f}.",
        "",
    ]

    passed = True
    for value in values:
        encoded = encode_real_to_binary(value, config)
        decoded = decode_binary_to_real(encoded, config)
        error = abs(value - decoded)
        lines.append(
            f"x_real={value:8.4f} -> x_bin={encoded} -> x_real_back={decoded:8.4f} | error={error:.6f}"
        )
        if error > max_quantization_error + 1e-12:
            passed = False

    return DebugScenarioResult(
        scenario_id="test_1",
        title="Real/Binary conversion",
        passed=passed,
        body="\n".join(lines),
    )


def run_debug_test_2() -> DebugScenarioResult:
    """
    test_2: validate tournament selection behavior
    """
    rng = random.Random(2026)
    costs = [120.0, 88.0, 95.0, 70.0, 110.0, 80.0]
    tournament_size = 3
    rounds_count = 8

    selected, rounds = tournament_selection_indices(
        costs=costs, tournament_size=tournament_size, rng=rng, num_parents=rounds_count
    )

    lines = [
        "Goal: validate tournament selection picks the best among sampled candidates.",
        f"Population costs: {costs}",
        f"Tournament size: {tournament_size}",
        "",
    ]

    passed = True
    for round_idx, (winner_idx, round_info) in enumerate(zip(selected, rounds), start=1):
        expected_winner = min(round_info.candidate_indices, key=lambda idx: costs[idx])
        ok = winner_idx == expected_winner
        passed = passed and ok
        candidates_fmt = ", ".join(
            f"{idx}(cost={cost:.1f})"
            for idx, cost in zip(round_info.candidate_indices, round_info.candidate_costs)
        )
        lines.append(
            f"Round {round_idx:02d}: [{candidates_fmt}] -> winner={winner_idx} "
            f"(expected={expected_winner}) {'OK' if ok else 'ERROR'}"
        )

    return DebugScenarioResult(
        scenario_id="test_2",
        title="Tournament selection operator",
        passed=passed,
        body="\n".join(lines),
    )


def run_debug_test_3() -> DebugScenarioResult:
    """
    test_3:
      a) show which individuals are selected as parents by p_cross
      b) for two selected individuals show crossover result
    """
    rng = random.Random(7)
    population = generate_population(pop_size=6, num_cities=8, rng=rng)
    p_cross = 0.60

    selection_result = select_parents_by_crossover_probability(
        population_size=len(population), p_cross=p_cross, rng=rng
    )

    lines = [
        "Goal: inspect parent selection by p_cross and resulting crossover.",
        f"Population size={len(population)}, p_cross={p_cross:.2f}",
        "",
        "a) Parent eligibility by crossover probability:",
    ]

    selected_set = set(selection_result.selected_indices)
    for idx, draw in enumerate(selection_result.draws):
        state = "SELECTED" if idx in selected_set else "NOT_SELECTED"
        lines.append(f"  individual {idx}: draw={draw:.4f} -> {state}")

    if selection_result.dropped_index is not None:
        lines.append(
            f"  odd number of selected individuals; dropped index={selection_result.dropped_index}."
        )

    passed = True
    lines.append("")
    lines.append("b) Crossover demo for first selected parent pair:")
    if len(selection_result.selected_indices) < 2:
        lines.append("  Not enough parents selected. Increase p_cross for this scenario.")
        passed = False
    else:
        p1_idx, p2_idx = selection_result.selected_indices[:2]
        parent_a = population[p1_idx]
        parent_b = population[p2_idx]
        child, debug = ordered_crossover(parent_a, parent_b, rng=rng)

        lines.append(f"  parent A (idx {p1_idx}): {parent_a}")
        lines.append(f"  parent B (idx {p2_idx}): {parent_b}")
        lines.append(f"  cuts: left={debug.cut_left}, right={debug.cut_right}")
        lines.append(f"  copied segment from A: {list(debug.copied_segment)}")
        lines.append(f"  fill order from B: {list(debug.fill_order_from_parent_b)}")
        lines.append(f"  child: {child}")

        valid = is_valid_permutation(child, num_cities=len(child))
        lines.append(f"  child is valid permutation: {valid}")
        passed = passed and valid

    return DebugScenarioResult(
        scenario_id="test_3",
        title="Parent selection + crossover trace",
        passed=passed,
        body="\n".join(lines),
    )


def run_debug_test_4() -> DebugScenarioResult:
    """
    test_4: show mutation performed on a specific individual
    """
    rng = random.Random(19)
    individual = list(range(10))
    mutated, debug = mutate_swap(individual, rng=rng)

    same_genes = sorted(mutated) == sorted(individual)
    changed = mutated != individual
    passed = same_genes and changed

    lines = [
        "Goal: show step-by-step mutation for one individual.",
        f"individual before: {list(debug.before)}",
        f"swap positions: i={debug.swap_i}, j={debug.swap_j}",
        f"individual after : {list(debug.after)}",
        f"same genes after mutation: {same_genes}",
        f"sequence changed: {changed}",
        f"valid permutation after mutation: {is_valid_permutation(mutated, len(mutated))}",
    ]

    return DebugScenarioResult(
        scenario_id="test_4",
        title="Mutation trace",
        passed=passed,
        body="\n".join(lines),
    )


DEBUG_SCENARIO_RUNNERS: dict[str, Callable[[], DebugScenarioResult]] = {
    "test_1": run_debug_test_1,
    "test_2": run_debug_test_2,
    "test_3": run_debug_test_3,
    "test_4": run_debug_test_4,
}


def run_debug_scenario(scenario_id: str) -> DebugScenarioResult:
    if scenario_id not in DEBUG_SCENARIO_RUNNERS:
        raise ValueError(f"Unknown debug scenario: {scenario_id}")
    return DEBUG_SCENARIO_RUNNERS[scenario_id]()

