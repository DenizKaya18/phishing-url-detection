#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run script for Classical ML Pipeline
Usage: python run.py
"""

import sys
import os

# Ensure main.py can be imported from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run main
if __name__ == "__main__":
    from main import main
    main()
