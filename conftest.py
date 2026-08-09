"""Make the project root importable so tests can `import data...` without an
editable install. pytest prepends the directory containing this file to sys.path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
