import random
import unittest

from tsp_ga import (
    BinaryEncodingConfig,
    decode_binary_to_real,
    encode_real_to_binary,
    is_valid_permutation,
    mutate_swap,
    ordered_crossover,
    select_parents_by_crossover_probability,
    tournament_selection_indices,
)


class TSPGATestCase(unittest.TestCase):
    def test_real_binary_roundtrip_within_quantization(self) -> None:
        config = BinaryEncodingConfig(min_value=0.0, max_value=100.0, bits=10)
        value = 37.42
        binary = encode_real_to_binary(value, config)
        decoded = decode_binary_to_real(binary, config)
        quantization_step = (config.max_value - config.min_value) / ((1 << config.bits) - 1)
        self.assertLessEqual(abs(decoded - value), quantization_step + 1e-12)

    def test_tournament_selection_picks_best_of_sample(self) -> None:
        rng = random.Random(2026)
        costs = [120.0, 88.0, 95.0, 70.0, 110.0, 80.0]
        selected, rounds = tournament_selection_indices(
            costs=costs, tournament_size=3, rng=rng, num_parents=10
        )
        for winner, round_info in zip(selected, rounds):
            expected = min(round_info.candidate_indices, key=lambda idx: costs[idx])
            self.assertEqual(winner, expected)

    def test_crossover_probability_parent_count_is_even(self) -> None:
        rng = random.Random(7)
        result = select_parents_by_crossover_probability(population_size=11, p_cross=0.63, rng=rng)
        self.assertEqual(len(result.selected_indices) % 2, 0)

    def test_ordered_crossover_produces_valid_permutation(self) -> None:
        rng = random.Random(11)
        parent_a = [0, 1, 2, 3, 4, 5, 6, 7]
        parent_b = [4, 2, 5, 1, 6, 7, 3, 0]
        child, _ = ordered_crossover(parent_a, parent_b, rng=rng)
        self.assertTrue(is_valid_permutation(child, num_cities=len(parent_a)))

    def test_mutation_swap_changes_route_but_keeps_genes(self) -> None:
        rng = random.Random(19)
        before = [0, 1, 2, 3, 4, 5, 6, 7]
        after, debug = mutate_swap(before, rng=rng)
        self.assertNotEqual(before, after)
        self.assertEqual(sorted(before), sorted(after))
        self.assertTrue(is_valid_permutation(after, num_cities=len(before)))
        self.assertEqual(list(debug.before), before)
        self.assertEqual(list(debug.after), after)


if __name__ == "__main__":
    unittest.main()

