"""Smoke test — verifies the drift package imports and version is present.

Expanded as real tests come online (judge tests in Unit 2.3, plug-in
contract test in Unit 0.3, etc.).
"""

import drift


def test_package_imports():
    assert drift.__version__ == "0.0.0"
