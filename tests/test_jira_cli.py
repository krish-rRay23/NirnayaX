"""Tests for Jira CLI subcommand and demo."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.cli import main


def test_cli_jira_demo_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()

    exit_code = main(
        [
            "jira-demo",
            "--train",
            str(tmp_path / "incidents_train.json"),
            "--runbooks",
            str(tmp_path / "runbooks.json"),
            "--model",
            str(tmp_path / "triage.joblib"),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Jira Integration + Human Approval Demo" in out
    assert "Jira Config" in out
    assert "Jira Audit Action Records" in out
    assert "Final Jira Issue Status: 'Resolved'" in out
