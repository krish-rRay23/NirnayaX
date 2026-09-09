"""Tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.cli import main


def test_generate_writes_all_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["generate", "--out", str(tmp_path), "--train-size", "80", "--eval-size", "30"]
    )
    assert exit_code == 0
    for name in (
        "runbooks.json",
        "incidents_train.json",
        "incidents_eval.json",
        "incidents_train.jsonl",
        "incidents_eval.jsonl",
    ):
        assert (tmp_path / name).exists(), name

    out = capsys.readouterr().out
    assert "TRAIN dataset" in out
    assert "EVAL dataset" in out
    assert "EDA summary" in out


def test_validate_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()  # clear
    exit_code = main(["validate", str(tmp_path / "incidents_train.json")])
    assert exit_code == 0
    assert "Validation: OK" in capsys.readouterr().out


def test_eda_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["generate", "--out", str(tmp_path), "--train-size", "50", "--eval-size", "20"])
    capsys.readouterr()
    exit_code = main(["eda", str(tmp_path / "incidents_train.json")])
    assert exit_code == 0
    assert "EDA summary" in capsys.readouterr().out


def test_seed_override_changes_train_data(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    common = ["--train-size", "40", "--eval-size", "10"]
    main(["generate", "--out", str(out_a), *common, "--seed", "111"])
    main(["generate", "--out", str(out_b), *common, "--seed", "222"])
    text_a = (out_a / "incidents_train.json").read_text(encoding="utf-8")
    text_b = (out_b / "incidents_train.json").read_text(encoding="utf-8")
    assert text_a != text_b
