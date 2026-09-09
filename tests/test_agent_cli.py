"""Tests for CLI diagnose subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.cli import main


def test_cli_diagnose_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Generate datasets and runbooks first
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()

    exit_code = main(
        [
            "diagnose",
            "--title",
            "BGP session flapping on edge router",
            "--description",
            (
                "BGP peering session is flapping between edge routers. "
                "Neighbor state stuck in active, routes withdrawn."
            ),
            "--service",
            "edge-router-1",
            "--train",
            str(tmp_path / "incidents_train.json"),
            "--runbooks",
            str(tmp_path / "runbooks.json"),
            "--model",
            str(tmp_path / "triage.joblib"),
            "--threshold",
            "0.50",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Agentic Diagnosis for Ticket" in out
    assert "Workflow Execution Log" in out
    assert "Final State:" in out
