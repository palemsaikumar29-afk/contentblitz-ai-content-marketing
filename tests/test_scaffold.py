"""Phase 1 smoke tests — the scaffold imports and versions cleanly."""

import contentblitz


def test_package_imports():
    assert contentblitz.__version__ == "0.1.0"
