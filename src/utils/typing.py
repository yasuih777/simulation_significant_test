# !/usr/bin/env python3

from typing import Callable, Literal, TypedDict


class TestInfo(TypedDict):
    """test information dictionary

    Attributes:
        method (Literal["normal", "welch", "student", "paired", "one-sample"]):
            significant test
        alternative (Literal["two-sided", "less", "greater"]):
            significant test type
        alpha (float): significance level
    """

    method: Literal["normal", "welch", "student", "paired", "one-sample"]
    alternative: Literal["two-sided", "less", "greater"]
    alpha: float


class GeneratorFunc(TypedDict):
    """generator function

    Attributes:
        rand (Callable): create sample function
        stat_prob (Callable): caluculate probability function
    """

    rand: Callable
    stat_prob: Callable
