# !/usr/bin/env python3

from typing import Callable, Literal, TypedDict


class TestInfo(TypedDict):
    method: Literal["normal", "welch", "student", "paired", "one-sample"]
    alternative: Literal["two-sided", "less", "greater"]
    alpha: float


class GeneratorFunc(TypedDict):
    rand: Callable
    stat_prob: Callable
