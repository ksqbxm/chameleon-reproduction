from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

import pytest

from scripts import regression_runner


ROOT = Path(__file__).parents[2]

@pytest.fixture
def local_tmp_path():
    root = ROOT / "artifacts" / "test-results"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runner-test-", dir=root) as directory:
        yield Path(directory)


def test_stage_output_is_written_incrementally(local_tmp_path, monkeypatch):
    tmp_path = local_tmp_path

    class Output:
        def __init__(self):
            self.first = True

        def __iter__(self):
            return self

        def __next__(self):
            if self.first:
                self.first = False
                return "first line\n"
            raise RuntimeError("injected output failure")

    class Process:
        stdout = Output()

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(regression_runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(regression_runner, "ROOT", tmp_path)
    report_dir = tmp_path / "artifacts" / "test-results"
    report_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="injected output failure"):
        regression_runner._run_stage(
            {"name": "stream", "world_size": None, "paths": ("tests/unit",)},
            "cpu", report_dir, tmp_path,
        )

    assert (report_dir / "final-cpu-stream.log").read_text(encoding="utf-8") == "first line\n"


def test_stage_exception_is_not_reported_as_preflight(local_tmp_path, monkeypatch):
    import chameleon.environment as environment

    tmp_path = local_tmp_path

    @contextmanager
    def execution_root():
        yield tmp_path

    monkeypatch.setattr(regression_runner, "ROOT", tmp_path)
    monkeypatch.setattr(regression_runner, "matrix", lambda _device: (
        {"name": "broken-stage", "world_size": None, "paths": ("tests/unit",)},
    ))
    monkeypatch.setattr(regression_runner, "_validate_coverage", lambda _stages: None)
    monkeypatch.setattr(regression_runner, "_execution_root", execution_root)
    monkeypatch.setattr(regression_runner, "_run_stage",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("injected stage failure")))
    monkeypatch.setattr(environment, "validate_device", lambda *_args: "gloo")

    assert regression_runner.run("cpu") == 1
    report = json.loads((tmp_path / "artifacts" / "test-results" /
                         "final-cpu-summary.json").read_text(encoding="utf-8"))
    assert report["failed_stage"] == "broken-stage"
    assert report["stage_error"] == "RuntimeError: injected stage failure"
    assert "preflight_error" not in report
