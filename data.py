#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def logistics(x):
    import numpy as _np
    return 1.0 / (1.0 + _np.exp(-x))

def generate_synthetic_nba_games(n=5000, seed=42):
    