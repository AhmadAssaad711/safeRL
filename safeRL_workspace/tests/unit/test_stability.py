"""Contract tests for the stability statistics.

These pin the arithmetic that decides whether a study's conclusion is reportable,
because the failure mode is silent: a wrong power formula or a mislabelled
variance component does not raise, it just makes an underpowered comparison look
decisive.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from saferl.stability import (
    DEFAULT_ALPHA,
    decile_summary,
    diagnostic_verdict,
    minimum_detectable_effect,
    paired_comparison,
    required_pairs,
    sd_confidence_interval,
    sign_test_p_value,
    tail_slope,
    variance_decomposition,
)

# The measured 1M state-cost study, so a regression in the statistics shows up as
# a changed verdict on real data rather than only on synthetic input.
CONTROL_MEANS = (3.683, 4.244, 3.209)
ELLIPSE_MEANS = (4.049, 4.345, 4.193)


class TestPairedComparison:
    def test_measured_study_is_not_significant(self) -> None:
        """The headline +13% must not read as significant at n=3."""

        pc = paired_comparison(CONTROL_MEANS, ELLIPSE_MEANS)
        assert pc.mean_difference == pytest.approx(0.484, abs=1e-3)
        assert pc.relative_difference == pytest.approx(0.130, abs=1e-3)
        assert pc.t_statistic == pytest.approx(1.85, abs=0.01)
        assert pc.p_value == pytest.approx(0.205, abs=0.005)
        assert not pc.significant

    def test_pairing_is_elementwise_not_pooled(self) -> None:
        """Differences must be taken within a seed, not between arm means."""

        pc = paired_comparison([1.0, 10.0], [2.0, 11.0])
        assert pc.differences.tolist() == [1.0, 1.0]
        assert pc.sd_difference == pytest.approx(0.0)

    def test_mismatched_lengths_and_tiny_samples_raise(self) -> None:
        with pytest.raises(ValueError):
            paired_comparison([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            paired_comparison([1.0], [2.0])


class TestSignTest:
    @pytest.mark.parametrize("n,expected", [(3, 0.25), (4, 0.125), (5, 0.0625)])
    def test_unanimous_agreement_hits_the_exact_floor(self, n: int, expected: float) -> None:
        """"Every seed agreed" cannot beat 2/2**n, so n=3 can never reach 0.05."""

        assert sign_test_p_value([1.0] * n) == pytest.approx(expected)
        assert sign_test_p_value([-1.0] * n) == pytest.approx(expected)

    def test_measured_study_is_at_the_floor(self) -> None:
        pc = paired_comparison(CONTROL_MEANS, ELLIPSE_MEANS)
        assert pc.sign_test_p_value == pytest.approx(0.25)

    def test_zero_differences_are_dropped(self) -> None:
        assert sign_test_p_value([0.0, 0.0]) == 1.0


class TestMinimumDetectableEffect:
    def test_half_power_is_the_bare_t_critical_form(self) -> None:
        """power=0.5 must reproduce t_crit * sd / sqrt(n) exactly."""

        # t(0.975, df=2) = 4.302653
        assert minimum_detectable_effect(0.453, 3, power=0.5) == pytest.approx(
            4.302653 * 0.453 / math.sqrt(3), rel=1e-4
        )

    def test_eighty_percent_power_is_strictly_larger(self) -> None:
        """The conventional planning number must exceed the coin-flip number.

        Conflating them understates the required seeds by roughly a third, which
        is exactly the error that makes an underpowered study look adequate.
        """

        for n in (3, 4, 6, 9, 12):
            half = minimum_detectable_effect(0.453, n, power=0.5)
            eighty = minimum_detectable_effect(0.453, n, power=0.8)
            assert eighty > half

    def test_default_power_is_eighty_percent(self) -> None:
        assert minimum_detectable_effect(0.453, 6) == pytest.approx(
            minimum_detectable_effect(0.453, 6, power=0.8)
        )

    def test_measured_study_resolves_only_a_large_effect(self) -> None:
        """n=3 at 80% power resolves ~38% relative, not the claimed 13%."""

        mde = minimum_detectable_effect(0.453, 3, power=0.8)
        assert mde == pytest.approx(1.401, abs=0.01)
        assert mde / 3.712 > 0.35

    def test_shrinks_with_n(self) -> None:
        values = [minimum_detectable_effect(0.453, n) for n in (3, 4, 6, 9, 20)]
        assert values == sorted(values, reverse=True)

    def test_rejects_impossible_power(self) -> None:
        with pytest.raises(ValueError):
            minimum_detectable_effect(0.453, 6, power=1.0)

    def test_degenerate_inputs_return_nan(self) -> None:
        assert math.isnan(minimum_detectable_effect(0.453, 1))
        assert math.isnan(minimum_detectable_effect(float("nan"), 6))


class TestRequiredPairs:
    def test_target_effect_drives_the_budget(self) -> None:
        """A 20% relative regression needs 6 paired seeds at the measured SD."""

        assert required_pairs(0.453, 0.742, power=0.8) == 6

    def test_conservative_sd_costs_more_seeds(self) -> None:
        conservative = math.sqrt(2) * 0.518
        assert required_pairs(conservative, 0.742, power=0.8) > required_pairs(
            0.453, 0.742, power=0.8
        )

    def test_smaller_target_needs_more_seeds(self) -> None:
        assert required_pairs(0.453, 0.3) > required_pairs(0.453, 0.9)

    def test_unreachable_target_reports_failure(self) -> None:
        assert required_pairs(0.453, 0.0) == -1
        assert required_pairs(0.453, 1e-6, max_pairs=10) == -1


class TestSdConfidenceInterval:
    def test_two_degrees_of_freedom_is_nearly_unbounded(self) -> None:
        """A 3-seed SD is too imprecise to plan from, and must read that way."""

        low, high = sd_confidence_interval(0.453, 3)
        assert low == pytest.approx(0.236, abs=0.005)
        assert high == pytest.approx(2.844, abs=0.01)
        assert high / low > 10

    def test_interval_tightens_with_observations(self) -> None:
        narrow = sd_confidence_interval(0.453, 50)
        wide = sd_confidence_interval(0.453, 3)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_interval_brackets_the_estimate(self) -> None:
        low, high = sd_confidence_interval(0.453, 10)
        assert low < 0.453 < high

    def test_degenerate_inputs_return_nan(self) -> None:
        assert all(math.isnan(v) for v in sd_confidence_interval(0.453, 1))


class TestVarianceDecomposition:
    def test_evaluation_noise_is_separated_from_seed_effect(self) -> None:
        """Identical policies differing only by evaluation noise show no seed effect."""

        rng = np.random.default_rng(307)
        runs = [rng.normal(4.0, 1.0, size=10) for _ in range(6)]
        result = variance_decomposition(runs)
        # Between-run spread here is pure evaluation noise, so the seed component
        # must stay small rather than absorbing it.
        assert result["seed_sd"] < result["evaluation_se_of_run_mean"] * 2.0

    def test_real_seed_effect_is_detected(self) -> None:
        rng = np.random.default_rng(308)
        runs = [rng.normal(mean, 0.2, size=10) for mean in (2.0, 4.0, 6.0, 8.0)]
        result = variance_decomposition(runs)
        assert result["seed_sd"] > 1.0
        assert result["seed_fraction_of_between_variance"] > 0.9

    def test_components_are_consistent(self) -> None:
        runs = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0])]
        result = variance_decomposition(runs)
        assert result["evaluation_se_of_run_mean"] == pytest.approx(
            result["within_run_block_sd"] / math.sqrt(4)
        )
        assert 0.0 <= result["seed_fraction_of_between_variance"] <= 1.0

    def test_seed_component_never_goes_negative(self) -> None:
        """Evaluation noise can exceed between-run spread; that is zero, not negative."""

        runs = [np.array([0.0, 10.0, 0.0, 10.0]), np.array([1.0, 9.0, 1.0, 9.0])]
        result = variance_decomposition(runs)
        assert result["seed_sd"] == 0.0

    def test_insufficient_input_raises(self) -> None:
        with pytest.raises(ValueError):
            variance_decomposition([np.array([1.0, 2.0])])
        with pytest.raises(ValueError):
            variance_decomposition([np.array([1.0]), np.array([2.0])])


class TestTailSlope:
    def test_recovers_a_known_slope_in_per_100k_units(self) -> None:
        steps = np.arange(0, 1_000_000, 1000, dtype=float)
        values = 100.0 + steps * (20.0 / 100_000.0)
        result = tail_slope(steps, values, tail_fraction=0.5)
        assert result.slope == pytest.approx(20.0, rel=1e-6)

    def test_flat_curve_reads_as_flat(self) -> None:
        rng = np.random.default_rng(309)
        steps = np.arange(0, 1_000_000, 1000, dtype=float)
        values = rng.normal(400.0, 5.0, size=steps.size)
        assert tail_slope(steps, values).is_flat()

    def test_rising_curve_does_not_read_as_flat(self) -> None:
        steps = np.arange(0, 1_000_000, 1000, dtype=float)
        values = 100.0 + steps * (20.0 / 100_000.0)
        assert not tail_slope(steps, values).is_flat()

    def test_tail_fraction_selects_only_the_tail(self) -> None:
        """A curve that plateaus must read flat once the early climb is excluded."""

        steps = np.arange(0, 1_000_000, 1000, dtype=float)
        values = np.where(steps < 500_000, steps / 1000.0, 500.0)
        assert tail_slope(steps, values, tail_fraction=0.4).slope == pytest.approx(
            0.0, abs=1e-6
        )

    def test_too_few_points_refuses_to_claim_certainty(self) -> None:
        result = tail_slope([0.0, 1.0], [0.0, 1.0], tail_fraction=1.0)
        assert math.isnan(result.t_statistic)
        assert result.is_flat()

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            tail_slope([0.0, 1.0], [0.0])


class TestDecileSummary:
    def test_averages_the_window_rather_than_the_endpoint(self) -> None:
        values = list(range(100))
        summary = decile_summary(values, fraction=0.1)
        assert summary["first_decile_mean"] == pytest.approx(4.5)
        assert summary["last_decile_mean"] == pytest.approx(94.5)
        assert summary["window"] == 10

    def test_short_series_keeps_at_least_one_point(self) -> None:
        summary = decile_summary([5.0], fraction=0.1)
        assert summary["first_decile_mean"] == 5.0
        assert summary["window"] == 1

    def test_empty_and_invalid_fraction_raise(self) -> None:
        with pytest.raises(ValueError):
            decile_summary([])
        with pytest.raises(ValueError):
            decile_summary([1.0], fraction=0.0)


class TestDiagnosticVerdict:
    def test_measured_runs_read_healthy(self) -> None:
        """The real 1M diagnostics must not be flagged; H1 is genuinely absent."""

        healthy, reasons = diagnostic_verdict(0.00886, 0.0838, 0.6061, 0.6397)
        assert healthy
        assert reasons == []

    def test_aggressive_updates_are_flagged(self) -> None:
        healthy, reasons = diagnostic_verdict(0.09, 0.08, 0.60, 0.60)
        assert not healthy
        assert any("approx_kl" in r for r in reasons)

    def test_dead_and_saturated_clipping_both_flagged(self) -> None:
        assert not diagnostic_verdict(0.009, 0.001, 0.6, 0.6)[0]
        assert not diagnostic_verdict(0.009, 0.55, 0.6, 0.6)[0]

    def test_exploration_collapse_is_flagged(self) -> None:
        healthy, reasons = diagnostic_verdict(0.009, 0.08, 0.60, 0.20)
        assert not healthy
        assert any("exploration" in r for r in reasons)

    def test_nan_inputs_do_not_manufacture_a_failure(self) -> None:
        healthy, _ = diagnostic_verdict(
            float("nan"), float("nan"), float("nan"), float("nan")
        )
        assert healthy
