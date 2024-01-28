# !/usr/bin/env python3

from collections import deque
from typing import Any, Literal, Optional

import numpy as np
from scipy import stats

from src.simulate.generate import DistGenerator
from src.utils.typing import TestFunc, TestInfo
from src.utils.utils import set_seed

TEST_TYPE = Literal["basic"]

class StatTestSimulator:
    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TEST_TYPE = "basic",
        seed: Optional[int] = None,
    ) -> None:
        self.test_funcs: TestFunc
        self.test_info = test_info
        self.test_type = test_type
        self.test_param: dict[str, Any] = {"alternative": test_info["alternative"]}

        self.generators = generators
        self.iters = iters
        self.results: dict[str, np.ndarray | float] = {
            "p_values": np.zeros(iters),
            "stat_values": np.zeros(iters),
            "parameter_value": np.zeros(iters),
        }

        self.h0_param: dict[str, int | float] = {}
        self.h1_param: dict[str, int | float] = {}
        self.rejection: list[float] = [-np.inf, np.inf]
        self.plot_prob = 0.9999

        set_seed(seed)

    def reset_h_parameter(self) -> None:
        raise NotImplementedError("Must override!!")

    def reset_refection(self):
        param = self.h0_param.copy()

        if self.test_info["alternative"] == "two-sided":
            param.update(q=self.test_info["alpha"] / 2)
            self.rejection[0] = self.test_funcs["stat_quan"](**param)
            self.rejection[1] = self.test_funcs["stat_quan_inv"](**param)
        elif self.test_info["alternative"] == "greater":
            param.update(q=self.test_info["alpha"])
            self.rejection[1] = self.test_funcs["stat_quan_inv"](**param)
        elif self.test_info["alternative"] == "less":
            param.update(q=self.test_info["alpha"])
            self.rejection[0] = self.test_funcs["stat_quan"](**param)

    def per_signicant(self) -> float:
        return (
            np.count_nonzero(self.results["p_values"] < self.test_info["alpha"])
            / self.iters
        )

    def calc_power(self) -> float:
        self.reset_h_parameter()
        self.reset_refection()

        param = self.h1_param.copy()
        param.update(x=self.rejection[0])
        lower = self.test_funcs["stat_dist"](**param)
        param.update(x=self.rejection[1])
        upper = self.test_funcs["stat_dist_inv"](**param)

        return lower + upper

    def basic_patch(self, idx: int) -> None:
        self.test_param.update(a=self.generators["X"].create_sample())
        if self.test_info["method"] != "one-sample":
            self.test_param.update(b=self.generators["Y"].create_sample())

        test_result = self.test_funcs["test"](**self.test_param)

        self.results["p_values"][idx] = test_result.pvalue
        self.results["stat_values"][idx] = test_result.statistic
        if "ttest" in self.test_funcs["test"].__name__:
            self.results["parameter_value"][idx] = test_result.df

    def execute(self) -> None:
        if self.test_type == "basic":
            batch_func = self.basic_patch
        else:
            pass

        deque(map(batch_func, range(self.iters)))

    def plot_range(self) -> tuple[float, float]:
        raise NotImplementedError("Must override!!")

    def density_points(self, delta: int = 100) -> dict[str, np.ndarray]:
        self.reset_h_parameter()
        self.reset_refection()

        h0_range = self.plot_range()
        h0_param = self.h0_param.copy()
        h1_param = self.h1_param.copy()

        h0_x = np.linspace(h0_range[0], h0_range[1], delta)
        h0_param.update(x=h0_x)
        h0_y = self.test_funcs["stat_prob"](**h0_param)
        h1_x = h0_x + np.mean(self.results["stat_values"])
        h1_param.update(x=h1_x)
        h1_y = self.test_funcs["stat_prob"](**h1_param)

        if len(h1_x[h1_x > self.rejection[1]]) > 0:
            h1_upperx = np.insert(h1_x[h1_x > self.rejection[1]], 0, self.rejection[1])
            h1_param.update(x=h1_upperx)
            h1_uppery = self.test_funcs["stat_prob"](**h1_param)
        else:
            h1_upperx, h1_uppery = np.array([]), np.array([])

        if len(h1_x[h1_x < self.rejection[0]]) > 0:
            h1_lowerx = np.append(h1_x[h1_x < self.rejection[0]], self.rejection[0])
            h1_param.update(x=h1_lowerx)
            h1_lowery = self.test_funcs["stat_prob"](**h1_param)
        else:
            h1_lowerx, h1_lowery = np.array([]), np.array([])

        return {
            "h0_x": h0_x,
            "h0_y": h0_y,
            "h1_x": h1_x,
            "h1_y": h1_y,
            "h1_upperx": h1_upperx,
            "h1_uppery": h1_uppery,
            "h1_lowerx": h1_lowerx,
            "h1_lowery": h1_lowery,
        }


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

        self.test_funcs = {
            "stat_prob": stats.nct.pdf,
            "stat_dist": stats.nct.cdf,
            "stat_dist_inv": stats.nct.sf,
            "stat_quan": stats.nct.ppf,
            "stat_quan_inv": stats.nct.isf,
        }

        if self.test_info["method"] == "welch":
            self.test_funcs["test"] = stats.ttest_ind
            self.test_param.update(equal_var=False)
        elif self.test_info["method"] == "student":
            self.test_funcs["test"] = stats.ttest_ind
        elif self.test_info["method"] == "paired":
            self.test_funcs["test"] = stats.ttest_rel
        elif self.test_info["method"] == "one-sample":
            self.test_funcs["test"] = stats.ttest_1samp
            self.test_param.update(popmean=mu)
        else:
            raise ValueError(
                "t test method must be [welch, student, paired, one-sample]"
            )

    def reset_h_parameter(self) -> None:
        df = np.mean(self.results["parameter_value"])
        nc = np.mean(self.results["stat_values"])

        self.h0_param.update(df=df, nc=0)
        self.h1_param.update(df=df, nc=nc)

    def plot_range(self) -> tuple[float, float]:
        param = self.h0_param.copy()
        param.update(confidence=self.plot_prob)

        return stats.nct.interval(**param)


def build_simulator(name: str, **args) -> StatTestSimulator:
    if name == "ttest":
        return TTestSimulator(**args)
    return StatTestSimulator(**args)
