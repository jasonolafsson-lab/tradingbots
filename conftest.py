"""
Pytest configuration.
Ensures the project root is in sys.path for imports.
"""
import sys
import os

# Add project root to path so tests can import all modules
sys.path.insert(0, os.path.dirname(__file__))
