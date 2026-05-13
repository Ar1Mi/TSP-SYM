import unittest

from debug_scenarios import run_debug_scenario


class DebugScenariosTestCase(unittest.TestCase):
    def test_all_debug_scenarios_pass(self) -> None:
        for scenario in ("test_1", "test_2", "test_3", "test_4"):
            result = run_debug_scenario(scenario)
            self.assertTrue(result.passed, msg=f"{scenario} should pass")
            self.assertIn(scenario, result.as_text())

    def test_unknown_scenario_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_debug_scenario("test_999")


if __name__ == "__main__":
    unittest.main()

