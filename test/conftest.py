import sys
from pathlib import Path
import asyncio


# Add the app/ directory to sys.path so "from gui import demo" resolves
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )