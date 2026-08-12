import unittest
from unittest import mock

import cent_power_calculator as cent


class DeviceRoleEnergyTest(unittest.TestCase):
    def setUp(self):
        self.stat = {"tCK_ps": 500.0}
        self.heads = 64
        self.hidden = 8192
        self.tokens = 129280
        self.gqa = 8
        self.phases = cent._analytical_dynamic_energy_by_operation(
            self.stat, self.heads, self.hidden, self.tokens, self.gqa
        )
        components = next(iter(self.phases.values())).keys()
        self.energy = {
            component: 1.0 + sum(phase[component] for phase in self.phases.values())
            for component in components
        }
        self.energy["PIM"] = 7.0
        self.latency = {
            "RMSNorm_latency": 1.0,
            "Softmax_latency": 2.0,
            "RotEmbed_latency": 3.0,
        }

    def calculate(self, role):
        with mock.patch.object(
            cent.CELLAR_POWER_CALCULATOR,
            "calculate_energy_and_latency",
            return_value=(dict(self.energy), dict(self.latency)),
        ):
            return cent.power_calculator(
                self.stat,
                1,
                self.heads,
                self.hidden,
                self.tokens,
                self.gqa,
                device_role=role,
            )

    def test_main_role_is_unchanged(self):
        energy, latency = self.calculate("main")
        self.assertEqual(energy, self.energy)
        self.assertEqual(latency, self.latency)

    def test_inter_device_helper_keeps_only_rotary_analytic_energy(self):
        energy, latency = self.calculate("inter_device_helper")
        for component in self.phases["RotEmbed"]:
            expected = 1.0 + self.phases["RotEmbed"][component]
            self.assertAlmostEqual(energy[component], expected)
        self.assertEqual(energy["PIM"], self.energy["PIM"])
        self.assertEqual(latency["RMSNorm_latency"], 0.0)
        self.assertEqual(latency["Softmax_latency"], 0.0)
        self.assertEqual(latency["RotEmbed_latency"], 3.0)

    def test_fc_helper_removes_all_three_analytic_operations(self):
        energy, latency = self.calculate("fc_helper")
        for component in self.phases["RotEmbed"]:
            self.assertAlmostEqual(energy[component], 1.0)
        self.assertEqual(energy["PIM"], self.energy["PIM"])
        self.assertTrue(all(value == 0.0 for value in latency.values()))

    def test_kv_head_helper_keeps_local_softmax_and_rotary(self):
        energy, latency = self.calculate("kv_head_helper")
        for component in self.phases["RotEmbed"]:
            expected = (
                1.0
                + self.phases["Softmax"][component]
                + self.phases["RotEmbed"][component]
            )
            self.assertAlmostEqual(energy[component], expected)
        self.assertEqual(energy["PIM"], self.energy["PIM"])
        self.assertEqual(latency["RMSNorm_latency"], 0.0)
        self.assertEqual(latency["Softmax_latency"], 2.0)
        self.assertEqual(latency["RotEmbed_latency"], 3.0)

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown device role"):
            self.calculate("not-a-role")


if __name__ == "__main__":
    unittest.main()
