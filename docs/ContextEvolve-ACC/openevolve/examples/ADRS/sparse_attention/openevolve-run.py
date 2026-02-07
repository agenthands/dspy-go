#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import torch
import torch.multiprocessing as mp

import sys
from openevolve.cli import main
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv()
    mp.set_start_method('spawn')
    sys.exit(main())