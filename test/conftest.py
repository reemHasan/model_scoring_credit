import sys
from pathlib import Path

# Add the app/ directory to sys.path so "from gui import demo" resolves
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))