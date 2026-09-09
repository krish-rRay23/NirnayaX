"""Integration tests for the ML CLI subcommands (train / evaluate / predict).

This is the acceptance path: a fresh dataset is generated, a model trained and
persisted, evaluated on a held-out split, and finally a brand-new ticket is
triaged end-to-end through the saved model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.cli import main


def _generate(tmp_path: Path) -> None:
    exit_code = main(
        ["generate", "--out", str(tmp_path), "--train-size", "300", "--eval-size", "120"]
    )
    assert exit_code == 0


def test_train_evaluate_predict_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generate(tmp_path)
    capsys.readouterr()  # clear generation output

    model_path = tmp_path / "triage.joblib"
    train_data = tmp_path / "incidents_train.json"
    eval_data = tmp_path / "incidents_eval.json"

    # --- train ---
    assert main(["train", "--train", str(train_data), "--out", str(model_path)]) == 0
    assert model_path.exists()
    assert (tmp_path / "triage.joblib.meta.json").exists()
    train_out = capsys.readouterr().out
    assert "model_version" in train_out

    # --- evaluate ---
    assert main(["evaluate", "--model", str(model_path), "--eval", str(eval_data)]) == 0
    eval_out = capsys.readouterr().out
    assert "Evaluation" in eval_out
    assert "category" in eval_out
    assert "confusion" in eval_out

    # --- predict a brand-new ticket ---
    assert (
        main(
            [
                "predict",
                "--model",
                str(model_path),
                "--title",
                "DNS resolution failures across region",
                "--description",
                "Users report NXDOMAIN and intermittent name resolution timeouts.",
                "--service",
                "resolver-fleet",
                "--channel",
                "MONITORING",
            ]
        )
        == 0
    )
    predict_out = capsys.readouterr().out
    assert "category" in predict_out
    assert "priority" in predict_out
    assert "consistent" in predict_out


def test_train_accepts_seed_and_reg_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generate(tmp_path)
    capsys.readouterr()
    model_path = tmp_path / "triage.joblib"
    exit_code = main(
        [
            "train",
            "--train",
            str(tmp_path / "incidents_train.json"),
            "--out",
            str(model_path),
            "--seed",
            "777",
            "--reg-c",
            "2.0",
        ]
    )
    assert exit_code == 0
    assert model_path.exists()
