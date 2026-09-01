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

    def test_accelerator_table_matches_cent_dev_7nm_model(self):
        self.assertNotIn("VEC", cent.ACCEL_POWER)
        expected = {
            "RED": (5.62e-02, 2.32e-01, 5.62e03),
            "EXP": (3.39e-02, 5.60e-01, 1.28e04),
            "VEC_ADD": (1.33e-01, 2.48e-01, 6.54e03),
            "VEC_MUL": (1.12e-01, 2.25e-01, 2.95e03),
            "TOPK": (2.92e-02, 7.47e-01, 3.77e03),
        }
        for name, (switch, internal, leak) in expected.items():
            power = cent.ACCEL_POWER[name]
            self.assertEqual(power["SWITCH"], switch)
            self.assertEqual(power["INT"], internal)
            self.assertEqual(power["LEAK"], leak)
            self.assertAlmostEqual(power["DYN"], switch + internal)
            self.assertAlmostEqual(power["STT"], leak / 1e6)
        self.assertEqual(cent.ACCEL_POWER["RV"], 3.96)

    def test_operation_terms_use_vec_add_not_legacy_vec(self):
        for operation in self.phases.values():
            self.assertIn("VEC_ADD_DYN", operation)
            self.assertNotIn("VEC_DYN", operation)

    def test_softmax_charges_two_vec_mul_operations(self):
        expected = (
            self.tokens
            * self.heads
            / 16.0
            * cent.ACCEL_POWER["VEC_MUL"]["DYN"]
            * 2.0
            * self.stat["tCK_ps"]
            / 1e12
        )
        self.assertAlmostEqual(
            self.phases["Softmax"]["VEC_MUL_DYN"], expected
        )
        self.assertEqual(self.phases["RMSNorm"]["VEC_MUL_DYN"], 0.0)
        self.assertEqual(self.phases["RotEmbed"]["VEC_MUL_DYN"], 0.0)

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
