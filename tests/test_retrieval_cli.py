"""Tests for retrieval CLI subcommands."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.cli import main


def test_cli_retrieve_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # First generate dataset and runbooks so files exist on disk
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()

    # Test basic retrieve
    exit_code = main(
        [
            "retrieve",
            "BGP peering session flapping",
            "--k",
            "3",
            "--runbooks",
            str(tmp_path / "runbooks.json"),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert 'query: "BGP peering session flapping"' in out
    assert "#1" in out

    # Test retrieve with category/subcategory filter and --no-rerank
    exit_code = main(
        [
            "retrieve",
            "BGP session flapping",
            "--k",
            "2",
            "--category",
            "NETWORK",
            "--subcategory",
            "BGP_ROUTING",
            "--no-rerank",
            "--runbooks",
            str(tmp_path / "runbooks.json"),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert 'query: "BGP session flapping"' in out


def test_cli_retrieval_eval_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()

    exit_code = main(
        [
            "retrieval-eval",
            "--train",
            str(tmp_path / "incidents_train.json"),
            "--eval",
            str(tmp_path / "incidents_eval.json"),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Retrieval [runbook]" in out
    assert "Retrieval [incident-similarity]" in out
