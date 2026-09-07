from scripts.ops.check_repository_policy import collect_violations


def test_repository_policy_has_no_structural_violations():
    assert collect_violations() == []
