from types import SimpleNamespace

import numpy as np

from scripts.evaluation.cbf_damped_gain_eval_sweep import (
    critical_damping_candidates,
    parse_c_values,
    physical_clearance_metrics,
)


def test_critical_damping_mapping_is_one_dimensional():
    candidates = critical_damping_candidates([0.5, 2.0, 8.0])

    assert [(row["c1"], row["c2"]) for row in candidates] == [(0.5, 0.5), (2.0, 2.0), (8.0, 8.0)]
    assert [row["k0"] for row in candidates] == [0.25, 4.0, 64.0]
    assert [row["k1"] for row in candidates] == [1.0, 4.0, 16.0]


def test_c_values_default_and_validation():
    assert parse_c_values(None) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    assert parse_c_values("0.5, 2, 8") == [0.5, 2.0, 8.0]

    for invalid in ("0", "-1", "1,1"):
        try:
            parse_c_values(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {invalid!r}")


def test_physical_clearance_uses_rectangle_collision_geometry():
    ego = SimpleNamespace(position=np.asarray([0.0, 5.0]), length=3.5, width=1.8)
    separated = SimpleNamespace(position=np.asarray([6.0, 5.0]), length=3.5, width=1.8)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            vehicle=ego,
            road=SimpleNamespace(vehicles=[ego, separated]),
            config={"road_width": 10.2},
            _signed_distance=lambda first, second: second - first,
        )
    )

    pairwise, boundary, overall = physical_clearance_metrics(env)

    assert pairwise == 2.5
    assert boundary == 4.1
    assert overall == 2.5
