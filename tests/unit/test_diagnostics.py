from labpilot.diagnostics import (
    CheckResult,
    check_environment,
    print_diagnostics_report,
    required_environment_checks,
)


def test_check_environment_returns_all_expected_checks():
    results = check_environment()
    names = {result.name for result in results}

    assert names == {
        "Python version",
        "LightGBM import",
        "Kaggle credentials",
        "LLM provider",
        "Image deps (torch/torchvision)",
        "Deep deps (torch/torchvision/transformers)",
    }
    assert all(isinstance(result, CheckResult) for result in results)


def test_check_environment_core_only_excludes_optional():
    results = check_environment(include_optional=False)
    names = {result.name for result in results}
    assert names == {"Python version", "LightGBM import", "Kaggle credentials", "LLM provider"}
    assert all(isinstance(result, CheckResult) for result in results)


def test_required_environment_checks_excludes_optional_and_can_skip_lightgbm():
    results = required_environment_checks(skip_lightgbm=True)
    names = {result.name for result in results}
    assert names == {"Python version", "Kaggle credentials", "LLM provider"}


def test_print_diagnostics_report_returns_true_only_when_all_ok(capsys):
    all_ok = [CheckResult("A", True, "fine"), CheckResult("B", True, "fine")]
    assert print_diagnostics_report(all_ok) is True

    one_bad = [CheckResult("A", True, "fine"), CheckResult("B", False, "broken", "fix it")]
    assert print_diagnostics_report(one_bad) is False


class _FakeOllama:
    """Stand-in for OllamaProvider with scriptable reachability/model list."""

    def __init__(self, reachable: bool, models: list[str]) -> None:
        self._reachable = reachable
        self._models = models

    def __call__(self, base_url, *args, **kwargs):  # constructed inside the check
        self.base_url = base_url
        return self

    def is_reachable(self, timeout_seconds: float = 2.0) -> bool:
        return self._reachable

    def list_models(self, timeout_seconds: float = 5.0) -> list[str]:
        return self._models


def _run_llm_check(monkeypatch, *, provider, model, reachable=True, models=None):
    import labpilot.llm.ollama as ollama_mod
    from labpilot.config import load_config
    from labpilot.diagnostics import _check_llm_provider

    config = load_config()
    config.llm.provider = provider
    config.llm.model = model
    # `load_config_for_cwd`, because that is what the check reads now — it used
    # to call `load_config()`, which sees only the package default, so inside a
    # workspace configuring a router `doctor` reported the legacy provider pin.
    # Patching `labpilot.config.load_config` no longer reaches it: `workspace.py`
    # binds the name at import, so the rebind misses and this helper would stub
    # nothing at all.
    monkeypatch.setattr(
        "labpilot.workspace.load_config_for_cwd", lambda *a, **k: (config, None)
    )
    monkeypatch.setattr(
        ollama_mod, "OllamaProvider", _FakeOllama(reachable, models or [])
    )
    return _check_llm_provider()


def test_llm_check_ok_when_ollama_has_model(monkeypatch):
    result = _run_llm_check(
        monkeypatch, provider="ollama", model="qwen2.5-coder:14b",
        models=["qwen2.5-coder:14b"],
    )
    assert result.ok is True
    assert "qwen2.5-coder:14b" in result.detail


def test_llm_check_fails_when_ollama_unreachable(monkeypatch):
    result = _run_llm_check(
        monkeypatch, provider="ollama", model="qwen2.5-coder:14b", reachable=False
    )
    assert result.ok is False
    assert "unreachable" in result.detail
    assert "ollama serve" in result.fix


def test_llm_check_fails_when_model_not_pulled(monkeypatch):
    result = _run_llm_check(
        monkeypatch, provider="ollama", model="missing:70b", models=["qwen2.5-coder:14b"]
    )
    assert result.ok is False
    assert "not pulled" in result.detail
    assert result.fix == "Run: ollama pull missing:70b"


def test_llm_check_accepts_implicit_latest_tag(monkeypatch):
    result = _run_llm_check(
        monkeypatch, provider="ollama", model="qwen2.5-coder", models=["qwen2.5-coder:latest"]
    )
    assert result.ok is True


def test_llm_check_flags_unknown_provider(monkeypatch):
    result = _run_llm_check(monkeypatch, provider="banana", model="x")
    assert result.ok is False
    assert "unknown provider" in result.detail
