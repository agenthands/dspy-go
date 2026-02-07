#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import sys
from openevolve.cli import main
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
