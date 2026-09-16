"""Statistics for deciding whether a training comparison can be believed.

Two different failures wear the same label, "the policy is not converging", and
they need different instruments:

**Optimizer instability.** The PPO update itself is misbehaving -- steps too
large, exploration collapsing, critic not fitting. This is visible in the
per-update diagnostics ``train/approx_kl``, ``train/clip_fraction``,
``train/explained_variance`` and ``train/std``, and it is what hyperparameter
tuning can fix.

**Variance swamping the effect.** The optimizer is healthy, but the reported KPI
carries so much noise -- across evaluation episodes, across seeds, and across
adjacent checkpoints of one run -- that the measured difference between two arms
is not distinguishable from nothing. No amount of tuning fixes this; only more
seeds, more episodes, or checkpoint averaging does.

The second failure is easy to mistake for a result. A paired comparison over
three seeds can only resolve an effect of roughly ``4.3 * sd / sqrt(3)``; if the
arms differ by less than that, "three of three seeds agree" is a statement about
a coin landing the same way three times. :func:`minimum_detectable_effect` makes
that threshold explicit *before* the effect is quoted, and
:func:`variance_decomposition` says how much of the seed-to-seed spread is real
policy difference versus evaluation noise that more episodes would remove.

Everything here is pure: arrays in, numbers out, no I/O and no plotting, so the
contract tests can pin the arithmetic independently of any run on disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats

# Conventional PPO health thresholds. These are reference points for reading a
# table, not pass/fail gates: a run is not broken because it crossed one.
APPROX_KL_CONCERN = 0.02
CLIP_FRACTION_DEAD = 0.01
CLIP_FRACTION_SATURATED = 0.30
DEFAULT_TAIL_FRACTION = 0.5
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
SLOPE_PER_STEPS = 100_000


@dataclass(frozen=True)
class SlopeResult:
    """Ordinary-least-squares trend of a training curve over its tail."""

    slope: float
    """Change in the metric per ``per_steps`` environment steps."""
    t_statistic: float
    p_value: float
    n_points: int
    per_steps: int

    @property
    def degrees_of_freedom(self) -> int:
        return max(self.n_points - 2, 0)

    def is_flat(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """Whether the tail trend is indistinguishable from zero.

        Not evidence of convergence on its own -- a curve can be flat because it
        converged or because it is too noisy to show its own slope. Read it with
        the KPI's spread, never alone.
        """

        return not math.isfinite(self.p_value) or self.p_value >= alpha


@dataclass(frozen=True)
class PairedComparison:
    """Paired (by seed) difference between a treatment arm and its control."""

    control_mean: float
    treatment_mean: float
    differences: np.ndarray = field(repr=False)
    mean_difference: float
    sd_difference: float
    standard_error: float
    t_statistic: float
    p_value: float
    n_pairs: int
    alpha: float

    @property
    def relative_difference(self) -> float:
        """Effect as a fraction of the control mean; NaN if the control is zero."""

        if self.control_mean == 0.0:
            return float("nan")
        return self.mean_difference / self.control_mean

    @property
    def significant(self) -> bool:
        return math.isfinite(self.p_value) and self.p_value < self.alpha

    @property
    def sign_test_p_value(self) -> float:
        """Exact two-sided sign test -- the "every seed agreed" claim, priced."""

        return sign_test_p_value(self.differences)

    @property
    def sd_confidence_interval(self) -> tuple[float, float]:
        """Interval for ``sd_difference``; wide at small ``n_pairs``."""

        return sd_confidence_interval(self.sd_difference, self.n_pairs, self.alpha)

    def minimum_detectable_effect(self, power: float = DEFAULT_POWER) -> float:
        """Smallest effect this design could have resolved, in KPI units.

        Defaults to 80% power, not the bare t-critical (50%) form.
        """

        return minimum_detectable_effect(
            self.sd_difference, self.n_pairs, self.alpha, power
        )


def decile_summary(
    values: Sequence[float] | np.ndarray, fraction: float = 0.1
) -> dict[str, float]:
    """Summarise a training curve by its first and last ``fraction`` of updates.

    Averaging a decile rather than reading the endpoints keeps the summary from
    being decided by a single noisy update.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("decile summary requires at least one value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    window = max(int(array.size * fraction), 1)
    head = array[:window]
    tail = array[-window:]
    return {
        "first_decile_mean": float(head.mean()),
        "last_decile_mean": float(tail.mean()),
        "last_decile_max": float(tail.max()),
        "last_decile_min": float(tail.min()),
        "overall_max": float(array.max()),
        "n_points": int(array.size),
        "window": int(window),
    }


def tail_slope(
    steps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    per_steps: int = SLOPE_PER_STEPS,
) -> SlopeResult:
    """Fit the trend of a training curve over its final ``tail_fraction``.

    A curve still climbing significantly at the budget means the comparison is
    measuring learning speed at an arbitrary cutoff rather than converged
    quality, so the sign of any effect may depend on where the budget fell.
    """

    x = np.asarray(steps, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"steps and values differ in length: {x.size} vs {y.size}")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in (0, 1]; got {tail_fraction}")
    order = np.argsort(x)
    x, y = x[order], y[order]
    window = max(int(x.size * tail_fraction), 2)
    x, y = x[-window:], y[-window:]
    if x.size < 3 or np.allclose(x, x[0]):
        # Fewer than three points leaves no residual degrees of freedom, so a
        # slope would come with no uncertainty and read as spuriously certain.
        return SlopeResult(float("nan"), float("nan"), float("nan"), int(x.size), per_steps)
    regression = stats.linregress(x, y)
    return SlopeResult(
        slope=float(regression.slope) * float(per_steps),
        t_statistic=float(regression.slope / regression.stderr)
        if regression.stderr
        else float("nan"),
        p_value=float(regression.pvalue),
        n_points=int(x.size),
        per_steps=int(per_steps),
    )


def minimum_detectable_effect(
    sd_difference: float,
    n_pairs: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest paired effect a design of ``n_pairs`` seeds could resolve.

    Note what ``power`` does, because getting it wrong overstates a design by
    about 50%. The familiar ``t_crit * sd / sqrt(n)`` is the effect that lands
    exactly on the significance boundary *in expectation*, i.e. the effect you
    would detect half the time -- ``power=0.5``. Detecting an effect reliably
    needs the second term:

        MDE = (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n)

    Default is ``DEFAULT_POWER`` (0.8), the conventional planning value. Pass
    ``power=0.5`` only to reproduce a bare t-critical table, and label it as
    such when you do.

    The target effect must be chosen on scientific grounds -- what size of
    regression would change the conclusion -- and never taken from the effect
    already observed, which is circular and always flatters the design.
    """

    if n_pairs < 2 or not math.isfinite(sd_difference):
        return float("nan")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1); got {power}")
    df = n_pairs - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    beta_term = 0.0 if power == 0.5 else float(stats.t.ppf(power, df=df))
    return (critical + beta_term) * float(sd_difference) / math.sqrt(n_pairs)


def required_pairs(
    sd_difference: float,
    target_effect: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    max_pairs: int = 200,
) -> int:
    """Smallest ``n`` whose detectable effect is no larger than ``target_effect``.

    ``target_effect`` is the difference worth detecting, decided in advance --
    not the difference already measured.
    """

    if target_effect <= 0.0 or not math.isfinite(sd_difference):
        return -1
    for n in range(2, int(max_pairs) + 1):
        if minimum_detectable_effect(sd_difference, n, alpha, power) <= target_effect:
            return n
    return -1


def sd_confidence_interval(
    sd: float, n_observations: int, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Confidence interval for a standard deviation estimated from ``n`` values.

    A paired SD from three seeds carries two degrees of freedom, and the
    resulting interval spans roughly a factor of ten. Any seed budget planned
    from such a point estimate is planned from noise, so report this interval
    next to every MDE derived from a small-sample SD.
    """

    df = int(n_observations) - 1
    if df < 1 or not math.isfinite(sd) or sd < 0.0:
        return (float("nan"), float("nan"))
    lower_chi2 = float(stats.chi2.ppf(1.0 - alpha / 2.0, df=df))
    upper_chi2 = float(stats.chi2.ppf(alpha / 2.0, df=df))
    return (
        sd * math.sqrt(df / lower_chi2),
        sd * math.sqrt(df / upper_chi2),
    )


def sign_test_p_value(differences: Sequence[float] | np.ndarray) -> float:
    """Exact two-sided sign-test p-value for "every seed agreed in direction".

    The floor matters: with ``n`` seeds all pointing the same way the smallest
    reachable two-sided p is ``2 / 2**n``, which is 0.25 at n=3 and 0.125 at
    n=4. So "three of three seeds against the term" cannot reach alpha=0.05
    under any test that respects the pairing, no matter how consistent it looks.
    """

    d = np.asarray(differences, dtype=float).reshape(-1)
    nonzero = d[d != 0.0]
    n = nonzero.size
    if n == 0:
        return 1.0
    positives = int((nonzero > 0).sum())
    k = min(positives, n - positives)
    tail = float(stats.binom.cdf(k, n, 0.5))
    return min(2.0 * tail, 1.0)


def paired_comparison(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> PairedComparison:
    """Paired t-test of ``treatment - control``, matched element-wise by seed.

    Pairing matters here: the same seed fixes the initial scenario and traffic,
    so differencing within a seed removes the scenario variance that otherwise
    dominates.
    """

    c = np.asarray(control, dtype=float).reshape(-1)
    t = np.asarray(treatment, dtype=float).reshape(-1)
    if c.size != t.size:
        raise ValueError(f"arms differ in length: {c.size} control vs {t.size} treatment")
    if c.size < 2:
        raise ValueError("a paired comparison requires at least two seeds")
    differences = t - c
    sd = float(differences.std(ddof=1))
    se = sd / math.sqrt(differences.size)
    if se > 0.0:
        t_stat = float(differences.mean() / se)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=differences.size - 1))
    else:
        t_stat = float("inf") if differences.mean() else 0.0
        p_value = 0.0 if differences.mean() else 1.0
    return PairedComparison(
        control_mean=float(c.mean()),
        treatment_mean=float(t.mean()),
        differences=differences,
        mean_difference=float(differences.mean()),
        sd_difference=sd,
        standard_error=float(se),
        t_statistic=t_stat,
        p_value=p_value,
        n_pairs=int(differences.size),
        alpha=float(alpha),
    )


def variance_decomposition(blocks_per_run: Sequence[Sequence[float] | np.ndarray]) -> dict[str, float]:
    """Split the spread between runs into evaluation noise and real seed effect.

    Each element of ``blocks_per_run`` is one run's per-block KPI values, where a
    block is an equal group of evaluation episodes. The variance of a run's mean
    from evaluation noise alone is the within-run block variance divided by the
    block count; whatever between-run variance remains after subtracting it is
    the genuine seed-to-seed component.

    The distinction decides what to spend on: an evaluation-dominated spread is
    fixed by more episodes, a seed-dominated one only by more seeds.
    """

    runs = [np.asarray(b, dtype=float).reshape(-1) for b in blocks_per_run]
    if len(runs) < 2:
        raise ValueError("variance decomposition requires at least two runs")
    if any(r.size < 2 for r in runs):
        raise ValueError("every run needs at least two blocks")
    within_var = float(np.mean([r.var(ddof=1) for r in runs]))
    block_counts = {r.size for r in runs}
    # Unequal block counts would make a single "evaluation SE" meaningless, so
    # use the mean count and say so rather than silently averaging apples.
    mean_blocks = float(np.mean([r.size for r in runs]))
    run_means = np.array([r.mean() for r in runs], dtype=float)
    between_var = float(run_means.var(ddof=1))
    evaluation_var = within_var / mean_blocks
    seed_var = max(between_var - evaluation_var, 0.0)
    return {
        "within_run_block_sd": math.sqrt(within_var),
        "evaluation_se_of_run_mean": math.sqrt(evaluation_var),
        "between_run_sd": math.sqrt(between_var),
        "seed_sd": math.sqrt(seed_var),
        "seed_fraction_of_between_variance": (seed_var / between_var) if between_var > 0 else 0.0,
        "n_runs": float(len(runs)),
        "mean_blocks_per_run": mean_blocks,
        "equal_block_counts": float(len(block_counts) == 1),
    }


def diagnostic_verdict(
    approx_kl_last: float, clip_fraction_last: float, std_first: float, std_last: float
) -> tuple[bool, list[str]]:
    """Flag the classic PPO optimizer pathologies in one run's tail.

    Returns ``(healthy, reasons)``. A healthy verdict only means these specific
    failures are absent; it is not a claim that the run converged.
    """

    reasons: list[str] = []
    if math.isfinite(approx_kl_last) and approx_kl_last > APPROX_KL_CONCERN:
        reasons.append(
            f"approx_kl {approx_kl_last:.4g} exceeds {APPROX_KL_CONCERN}: updates too aggressive"
        )
    if math.isfinite(clip_fraction_last):
        if clip_fraction_last < CLIP_FRACTION_DEAD:
            reasons.append(
                f"clip_fraction {clip_fraction_last:.4g} near zero: policy barely changing"
            )
        elif clip_fraction_last > CLIP_FRACTION_SATURATED:
            reasons.append(
                f"clip_fraction {clip_fraction_last:.4g} high: pinned to the trust region"
            )
    if math.isfinite(std_first) and math.isfinite(std_last) and std_first > 0:
        if std_last / std_first < 0.5:
            reasons.append(
                f"policy std fell {std_first:.4g} -> {std_last:.4g}: exploration collapsing"
            )
    return (not reasons), reasons
