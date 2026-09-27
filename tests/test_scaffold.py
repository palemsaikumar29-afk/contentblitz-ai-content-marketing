"""Package smoke tests."""
import contentblitz


def test_package_imports():
    assert contentblitz.__version__ == "0.2.0"
    for mod in ("agents", "brand", "config", "exports", "graph", "memory",
                "quality", "tools", "workflows"):
        assert mod in dir(contentblitz), mod
