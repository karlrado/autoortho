#!/usr/bin/env python
"""Test script to verify PyQt6 UI loads correctly"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from autoortho.aoconfig import AOConfig
from autoortho.config_ui_qt import ConfigUI

def test_ui():
    """Test the PyQt6 UI"""
    app = QApplication(sys.argv)

    # Load config
    cfg = AOConfig()

    # Create and show UI
    ui = ConfigUI(cfg)
    ui.show()

    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    test_ui()
