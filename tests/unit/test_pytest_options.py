import subprocess
import sys

import pytest


@pytest.mark.parametrize("arguments,expected", [
    (["--world-size", "0"], "--world-size must be >= 1"),
    (["--device", "cpu", "--require-gpu"], "--require-gpu requires --device cuda"),
    (["--device", "gpu"], "invalid choice"),
])
def test_invalid_pytest_options_fail(arguments, expected):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_contracts.py", "--collect-only", *arguments],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 4
    assert expected in result.stderr


@pytest.mark.parametrize("arguments,requested,generic,recovery", [
    ([], None, 2, 4),
    (["--world-size", "4"], 4, 4, 4),
    (["--world-size=2"], 2, 2, 2),
    (["--world-size", "8"], 8, 8, 8),
])
def test_world_size_parsing_preserves_explicit_requests(arguments, requested, generic, recovery):
    import json
    code = """
import json
import sys
import pytest
class Audit:
    def pytest_collection_finish(self, session):
        from conftest import _world_size
        config = session.config
        print('worker-sizes=' + json.dumps([
            config.getoption('--world-size'), _world_size(config), _world_size(config, default=4)]))
sys.exit(pytest.main(['tests/unit/test_contracts.py', '--collect-only', '-q', *sys.argv[1:]], plugins=[Audit()]))
"""
    result = subprocess.run([sys.executable, "-c", code, *arguments],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker-sizes=" + json.dumps([requested, generic, recovery]) in result.stdout


@pytest.mark.parametrize("size", [2, 3, 5])
def test_recovery_entry_rejects_explicit_wrong_size_before_startup(size):
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/distributed/test_full_state_transfer.py::test_complete_model_and_adamw_hashes_survive_missing_only_p2p",
        "--device", "cpu", "--world-size", str(size), "-q", "--tb=short",
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert "Task12 DP2/PP2 acceptance requires --world-size 4" in result.stdout
    assert "ModuleNotFoundError" not in result.stdout
