import sys

import pytest
import typer

from labpilot.cli import main as cli_main
from labpilot.config import LLMConfig


class FakeLLMClient:
    def complete(self, system: str, user: str) -> str:
        return "unused"


class FakeStdin:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def _no_client_config() -> LLMConfig:
    # No API key -> `create_llm_client` really does return `None`, so these
    # tests exercise the actual "no LLM available" branch end to end rather
    # than mocking `create_llm_client` itself.
    return LLMConfig(provider="openai", api_key="")


def test_check_llm_or_confirm_returns_client_immediately_without_prompting(monkeypatch):
    fake_client = FakeLLMClient()
    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda config: fake_client)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Confirm.ask should not be called when a client is available")

    monkeypatch.setattr(cli_main.Confirm, "ask", _fail_if_called)

    result = cli_main._check_llm_or_confirm(LLMConfig(provider="openai", api_key="sk-test"), False)

    assert result is fake_client


def test_check_llm_or_confirm_auto_proceeds_with_assume_yes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", FakeStdin(is_tty=True))
    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda config: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Confirm.ask should not be called when --yes is passed")

    monkeypatch.setattr(cli_main.Confirm, "ask", _fail_if_called)

    result = cli_main._check_llm_or_confirm(_no_client_config(), True)

    assert result is None
    assert "Proceeding without LLM" in capsys.readouterr().out


def test_check_llm_or_confirm_auto_proceeds_when_non_interactive(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", FakeStdin(is_tty=False))
    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda config: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Confirm.ask should not be called for non-interactive runs")

    monkeypatch.setattr(cli_main.Confirm, "ask", _fail_if_called)

    result = cli_main._check_llm_or_confirm(_no_client_config(), False)

    assert result is None
    assert "Proceeding without LLM" in capsys.readouterr().out


def test_check_llm_or_confirm_returns_none_when_user_confirms(monkeypatch):
    monkeypatch.setattr(sys, "stdin", FakeStdin(is_tty=True))
    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda config: None)
    monkeypatch.setattr(cli_main.Confirm, "ask", lambda *args, **kwargs: True)

    result = cli_main._check_llm_or_confirm(_no_client_config(), False)

    assert result is None


def test_check_llm_or_confirm_exits_when_user_declines(monkeypatch):
    monkeypatch.setattr(sys, "stdin", FakeStdin(is_tty=True))
    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda config: None)
    monkeypatch.setattr(cli_main.Confirm, "ask", lambda *args, **kwargs: False)

    with pytest.raises(typer.Exit):
        cli_main._check_llm_or_confirm(_no_client_config(), False)
