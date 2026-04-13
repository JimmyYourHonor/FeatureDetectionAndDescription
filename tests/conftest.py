import sys
import os
from unittest.mock import MagicMock

# Make the project root importable when running pytest from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# wandb is not required for testing and may not be installed in the test
# environment.  Mock the entire module before any project code imports it.
sys.modules.setdefault("wandb", MagicMock())
