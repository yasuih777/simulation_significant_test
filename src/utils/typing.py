# !/usr/bin/env python3

from typing import Callable, Literal, Optional, TypedDict


class TestInfo(TypedDict):
    method: Optional[Literal["", "welch", "student", "paired", "one-sample"]]
    alternative: Literal["two-sided", "less", "greater"]
    alpha: float


class GeneratorFunc(TypedDict):
    rand: Callable
    stat_prob: Callable


class TestFunc(TypedDict):
    test: Callable
    stat_prob: Callable
    stat_dist: Callable
    stat_dist_inv: Callable
    stat_quan: Callable
    stat_quan_inv: Callable
