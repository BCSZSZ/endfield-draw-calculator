import unittest

import numpy as np

from draw_model import character_distributions
from joint_model import _free_ten, build_joint_history


class JointRuleTests(unittest.TestCase):
    def test_free_ten_earns_quota_without_touching_normal_pity(self):
        state = np.zeros((1, 1, 2, 7, 1))
        state[0, 0, 0, 0, 0] = 1.0
        result = _free_ten(state)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertAlmostEqual(float(result[:, :, 1].sum()), 0.0)
        self.assertAlmostEqual(float(result[..., 0].sum()), 0.0)  # At least one >=5-star.
        distribution = result.sum(axis=(0, 1, 2, 4))
        self.assertAlmostEqual(distribution[0], 0.996**10)

    def test_one_draw_expected_quota_and_joint_probability(self):
        from weapon_draw import calculate_weapon_full_potential

        joint = build_joint_history(1)
        weapon = calculate_weapon_full_potential(1, 1)['finish_probs'][1]
        self.assertAlmostEqual(joint.metrics[1, 1], 50.24)
        self.assertAlmostEqual(joint.probabilities[1, 1, 1], 0.004 * weapon)

    def test_quota_requires_a_full_1980_tickets(self):
        from joint_character_weapon_dp import _weapon_ten_pulls_from_quota_units

        self.assertEqual(_weapon_ten_pulls_from_quota_units(98), 0)
        self.assertEqual(_weapon_ten_pulls_from_quota_units(99), 1)

    def test_character_marginal_matches_exact_model(self):
        joint = build_joint_history(121, tail_tolerance=1e-14)
        character = character_distributions(121)
        tails = character[:, ::-1].cumsum(axis=1)[:, ::-1]
        np.testing.assert_allclose(joint.probabilities[:, :, 0], tails, atol=2e-11)
        np.testing.assert_allclose(joint.metrics[:, 4], 1.0, atol=2e-11)
        self.assertGreater(joint.metrics[30, 1], joint.metrics[29, 1] + 200)
        self.assertLessEqual(joint.metrics[120, 0] - joint.metrics[119, 0], 1.0)
        self.assertLessEqual(joint.metrics[120, 1] - joint.metrics[119, 1], 2000.0)
        self.assertTrue(np.all(joint.probabilities[:, 1:, 1:] <= joint.probabilities[:, :1, 1:] + 1e-12))


if __name__ == '__main__':
    unittest.main()
