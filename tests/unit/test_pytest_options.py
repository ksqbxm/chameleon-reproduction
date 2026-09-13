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
