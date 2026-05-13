import unittest

from tsp_solver import GAConfig, bundled_instance_names, load_instance_by_name, run_ga


class TSPSolverIntegrationTestCase(unittest.TestCase):
    def test_bundled_instances_resolve(self) -> None:
        names = bundled_instance_names()
        self.assertGreaterEqual(len(names), 4)
        instance = load_instance_by_name(names[0])
        self.assertGreaterEqual(instance.dimension, 2)
        self.assertEqual(len(instance.cities), instance.dimension)

    def test_run_ga_returns_consistent_histories(self) -> None:
        instance = load_instance_by_name("eil51.tsp")
        config = GAConfig(
            population_size=40,
            num_generations=30,
            selection_method="tournament",
            tournament_size=3,
            crossover_method="ox",
            crossover_prob=0.9,
            mutation_method="2-swap",
            mutation_prob=0.05,
            elitist=True,
            hillclimbing=False,
            seed=42,
        )
        result = run_ga(instance, config)
        self.assertEqual(len(result.history_best), config.num_generations + 1)
        self.assertEqual(len(result.history_avg), config.num_generations + 1)
        self.assertEqual(len(result.best_route), instance.dimension)
        self.assertEqual(set(result.best_route), set(range(instance.dimension)))
        self.assertLessEqual(min(result.history_best), result.history_best[0] + 1e-12)

    def test_hillclimbing_variant_executes(self) -> None:
        instance = load_instance_by_name("berlin52.tsp")
        config = GAConfig(
            population_size=30,
            num_generations=10,
            selection_method="my-roulette",
            tournament_size=3,
            crossover_method="pmx",
            crossover_prob=0.85,
            mutation_method="inversion",
            mutation_prob=0.10,
            elitist=True,
            hillclimbing=True,
            hillclimbing_variant="3-opt",
            hillclimbing_start_generation=0,
            seed=7,
        )
        result = run_ga(instance, config)
        self.assertEqual(result.snapshots[-1].generation, config.num_generations)
        self.assertTrue(result.best_cost > 0.0)


if __name__ == "__main__":
    unittest.main()

