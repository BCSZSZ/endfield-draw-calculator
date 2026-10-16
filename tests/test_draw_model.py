import unittest

import numpy as np

from character_markov import _get_rates


class CharacterRuleTests(unittest.TestCase):
    def test_uncovered_quantile_is_not_a_fabricated_pull_count(self):
        from draw_model import distribution_statistics

        result = distribution_statistics(np.array([0.0, 0.25]))
        self.assertIsNone(result['p50'])
        self.assertEqual(result['mean'], 1.0)
        self.assertEqual(result['variance'], 0.0)

    def test_soft_pity_starts_on_pull_66(self):
        self.assertAlmostEqual(sum(_get_rates(65, 80)[:2]), 0.058)

    def test_pull_80_is_guaranteed_six_star(self):
        self.assertEqual(_get_rates(79, 80), (0.5, 0.5, 0.0))

    def test_free_up_preserves_first_up_guarantee(self):
        from draw_model import character_step

        state = np.zeros((80, 2, 7))
        state[15, 0, 1] = 1.0  # A free UP, no ordinary UP yet.
        result = character_step(state, 120)
        self.assertAlmostEqual(result[0, 1, 2], 1.0)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_free_pulls_preserve_both_pity_states(self):
        from draw_model import add_free_pulls

        state = np.zeros((80, 2, 7))
        state[42, 0, 0] = 1.0
        result = add_free_pulls(state)
        self.assertAlmostEqual(result.sum(), 1.0)
        self.assertAlmostEqual(result[:, 1, :6].sum(), 0.0)
        self.assertAlmostEqual(result[42, 0, :6].sum(), result[:, :, :6].sum())

    def test_120_guarantee_is_consumed_by_an_ordinary_up(self):
        from draw_model import character_step

        state = np.zeros((80, 2, 7))
        state[10, 1, 1] = 1.0
        self.assertAlmostEqual(character_step(state, 120)[:, :, 2].sum(), 0.004)

    def test_240_gift_adds_copy_without_resetting_pity(self):
        from draw_model import character_step

        state = np.zeros((80, 2, 7))
        state[42, 1, 0] = 1.0
        self.assertAlmostEqual(character_step(state, 239)[:, :, 0].sum(), 0.996)
        gifted = character_step(state, 240)
        self.assertAlmostEqual(gifted[43, 1, 1], 0.992)
        self.assertAlmostEqual(gifted[:, :, 2].sum(), 0.004)

    def test_banners_conserve_mass_and_guarantee_targets(self):
        from draw_model import character_distributions

        for banner in ('chartered', 'rerun'):
            distributions = character_distributions(1200, banner=banner)
            np.testing.assert_allclose(distributions.sum(axis=1), 1.0, atol=2e-13)
            self.assertAlmostEqual(distributions[120, 0], 0.0)
            self.assertAlmostEqual(distributions[1200, 6], 1.0)
            tails = distributions[:, ::-1].cumsum(axis=1)[:, ::-1]
            self.assertGreaterEqual(float(np.diff(tails[:, 1:], axis=0).min()), -1e-13)

    def test_rerun_extra_free_pulls_trigger_at_60_and_90(self):
        from draw_model import character_distributions

        first = character_distributions(90)
        rerun = character_distributions(90, banner='rerun')
        np.testing.assert_allclose(first[:60], rerun[:60])
        self.assertLess(rerun[60, 0], first[60, 0])
        self.assertLess(rerun[90, 0], first[90, 0])


if __name__ == '__main__':
    unittest.main()
