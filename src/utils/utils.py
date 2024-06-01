# !/usr/bin/env python3

import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int]) -> None:
    """set seed

    Args:
        seed (Optional[int]): fixed seed number
    """
    random.seed(seed)
    np.random.seed(seed)
