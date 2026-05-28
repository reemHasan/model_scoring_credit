import sys
from pathlib import Path

# Add the app/ directory to sys.path so "from gui import demo" resolves
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

# Pre-set the state_store so gui.py works in tests too
from app.state_store import set_state

def pytest_configure(config):
    """Called before any test — initialise state_store with a dummy."""
    pass  # state is set per-fixture, not globally