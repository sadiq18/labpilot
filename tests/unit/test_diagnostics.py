from labpilot.diagnostics import CheckResult, check_environment, print_diagnostics_report


def test_check_environment_returns_all_expected_checks():
    results = check_environment()
    names = {result.name for result in results}

    assert names == {
        "Python version",
        "LightGBM import",
        "Kaggle credentials",
        "Image deps (torch/torchvision)",
        "Deep deps (torch/torchvision/transformers)",
    }
    assert all(isinstance(result, CheckResult) for result in results)


def test_check_environment_core_only_excludes_optional():
    results = check_environment(include_optional=False)
    names = {result.name for result in results}
    assert names == {"Python version", "LightGBM import", "Kaggle credentials"}
    assert all(isinstance(result, CheckResult) for result in results)


def test_print_diagnostics_report_returns_true_only_when_all_ok(capsys):
    all_ok = [CheckResult("A", True, "fine"), CheckResult("B", True, "fine")]
    assert print_diagnostics_report(all_ok) is True

    one_bad = [CheckResult("A", True, "fine"), CheckResult("B", False, "broken", "fix it")]
    assert print_diagnostics_report(one_bad) is False
