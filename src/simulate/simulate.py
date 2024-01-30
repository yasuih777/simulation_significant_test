# !/usr/bin/env python3

from collections import deque
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy import stats

from src.simulate.generate import DistGenerator
from src.utils.typing import TestInfo
from src.utils.utils import set_seed

TEST_TYPE = Literal["basic", "another_test"]


"""
Basic simulator
"""
class StatTestSimulator:
    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TEST_TYPE = "basic",
        seed: Optional[int] = None,
    ) -> None:
        self.test_funcs: Callable
        self.test_info = test_info
        self.test_type = test_type
        self.test_param: dict[str, Any] = {"alternative": test_info["alternative"]}

        self.generators = generators
        self.iters = iters
        self.p_values = np.zeros(iters)

        set_seed(seed)

    def per_signicant(self) -> float:
        return np.count_nonzero(self.p_values < self.test_info["alpha"]) / self.iters

    def basic_patch(self, idx: int) -> None:
        self.sample_update()
        test_result = self.test_funcs(**self.test_param)
        self.p_values[idx] = test_result.pvalue
    
    def another_test_patch(self, idx: int) -> None:
        for _ in range(2):
            self.sample_update()
            test_result = self.test_funcs(**self.test_param)
            pvalue = test_result.pvalue
            if pvalue < self.test_info["alpha"]:
                break
        self.p_values[idx] = pvalue

    def execute(self) -> None:
        if self.test_type == "basic":
            batch_func = self.basic_patch
        elif self.test_type == "another_test":
            batch_func = self.another_test_patch
        else:
            pass

        deque(map(batch_func, range(self.iters)))

    def sample_update(self) -> None:
        raise NotImplementedError("Must override!!")

"""
2 samples simulator
"""
class TTestSimulator(StatTestSimulator):
    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TEST_TYPE = "basic",
        seed: int | None = None,
        mu: float = 0,
    ) -> None:
        super().__init__(test_info, generators, iters, test_type, seed)

        if self.test_info["method"] == "welch":
            self.test_funcs = stats.ttest_ind
            self.test_param.update(equal_var=False)
        elif self.test_info["method"] == "student":
            self.test_funcs = stats.ttest_ind
        elif self.test_info["method"] == "paired":
            self.test_funcs = stats.ttest_rel
        elif self.test_info["method"] == "one-sample":
            self.test_funcs = stats.ttest_1samp
            self.test_param.update(popmean=mu)
        else:
            raise ValueError(
                "t test method must be [welch, student, paired, one-sample]"
            )

    def sample_update(self) -> None:
        self.test_param.update(a=self.generators["X"].create_sample())
        if self.test_info["method"] != "one-sample":
            self.test_param.update(b=self.generators["Y"].create_sample())


class WilcoxonTestSimulator(StatTestSimulator):
    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TEST_TYPE = "basic",
        seed: int | None = None,
    ) -> None:
        super().__init__(test_info, generators, iters, test_type, seed)

        if self.test_info["method"] == "normal":
            self.test_funcs = stats.mannwhitneyu
        elif self.test_info["method"] == "paired":
            self.test_funcs = stats.wilcoxon
        else:
            raise ValueError("t test method must be [normal, paired]")

    def sample_update(self) -> None:
        self.test_param.update(x=self.generators["X"].create_sample())
        self.test_param.update(y=self.generators["Y"].create_sample())


class BrunnerMunzelTestSimulator(StatTestSimulator):
    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TEST_TYPE = "basic",
        seed: int | None = None,
    ) -> None:
        super().__init__(test_info, generators, iters, test_type, seed)

        self.test_funcs = stats.brunnermunzel

    def sample_update(self) -> None:
        self.test_param.update(x=self.generators["X"].create_sample())
        self.test_param.update(y=self.generators["Y"].create_sample())


def build_simulator(name: str, **args) -> StatTestSimulator:
    if name == "t_test":
        return TTestSimulator(**args)
    elif name == "wilcoxon_test":
        return WilcoxonTestSimulator(**args)
    elif name == "brunner_munzel_test":
        return BrunnerMunzelTestSimulator(**args)
    return StatTestSimulator(**args)
