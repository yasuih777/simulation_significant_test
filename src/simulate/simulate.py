# !/usr/bin/env python3

from collections import deque
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy import stats

from src.simulate.generate import DistGenerator
from src.utils.typing import TestInfo
from src.utils.utils import set_seed

TESTTYPE = Literal["basic", "another_test", "add_sample"]


class StatTestSimulator:
    """basic simulatior

    Attributes:
        test_funcs (Callable): significant test function
        test_info (TestInfo): significant test parameters
        test_type (TESTTYPE): significant test operation
        test_param (dict[str, Any]): significant test operation parameters
        generators (dict[str, DistGenerator]): distribution generators
        iters (int): simulation iteration
        p_values (np.ndarray): simulation pvalues, dim is (iters, )
    """

    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TESTTYPE = "basic",
        seed: Optional[int] = None,
    ) -> None:
        """initiation

        Args:
            test_info (TestInfo): significant test parameters
            generators (dict[str, DistGenerator]): distribution generators
            iters (int, optional): simulation iteration. Defaults to 10000.
            test_type (TESTTYPE, optional): significant test operation. Defaults to "basic".
            seed (Optional[int], optional): fixed seed number. Defaults to None.
        """
        self.test_funcs: Callable
        self.test_info = test_info
        self.test_type = test_type
        self.test_param: dict[str, Any] = {"alternative": test_info["alternative"]}

        self.generators = generators
        self.iters = iters
        self.p_values = np.zeros(iters)

        set_seed(seed)

    def per_signicant(self) -> float:
        """calculate percentage of p value belows significance level

        Returns:
            float: percentage of p value belows significance level
        """
        return np.count_nonzero(self.p_values < self.test_info["alpha"]) / self.iters

    def basic_patch(self, idx: int) -> None:
        """calculate p value by basic simulation

        Args:
            idx (int): simulation index
        """
        self.sample_update()
        test_result = self.test_funcs(**self.test_param)
        self.p_values[idx] = test_result.pvalue

    def another_test_patch(self, idx: int) -> None:
        """calculate p value by another simulation

        Note:
            - if p-value aboves significance level,
                generate another sample and perform a significant test.

        Args:
            idx (int): simulation index
        """
        for _ in range(2):
            self.sample_update()
            test_result = self.test_funcs(**self.test_param)
            pvalue = test_result.pvalue
            if pvalue < self.test_info["alpha"]:
                break
        self.p_values[idx] = pvalue

    def add_sample_patch(self, idx: int, add_sample_ratio: float = 1.1) -> None:
        """calculate p value by add sample simulation

        Notes:
            - if p-value aboves significance level,
                add sample and perform a significant test.

        Args:
            idx (int): simulation index
            add_sample_ratio (float, optional): add sample ratio from sample size.
                Defaults to 1.1.
        """
        sample_size = {}
        for key, generator in self.generators.items():
            sample_size[key] = generator.sample_size

        self.sample_update()
        test_result = self.test_funcs(**self.test_param)
        if test_result.pvalue >= self.test_info["alpha"]:
            for generator in self.generators.values():
                generator.sample_size = int(add_sample_ratio * generator.sample_size)

            self.sample_update()
            test_result = self.test_funcs(**self.test_param)

            for key, generator in self.generators.items():
                generator.sample_size = sample_size[key]

        self.p_values[idx] = test_result.pvalue

    def execute(self) -> None:
        """execute simulations"""
        batch_func: Callable
        if self.test_type == "basic":
            batch_func = self.basic_patch
        elif self.test_type == "another_test":
            batch_func = self.another_test_patch
        elif self.test_type == "add_sample":
            batch_func = self.add_sample_patch
        else:
            pass

        deque(map(batch_func, range(self.iters)))

    def sample_update(self) -> None:
        """update sample

        Raises:
            NotImplementedError: must override
        """
        raise NotImplementedError("Must override!!")


class TTestSimulator(StatTestSimulator):
    """T test simulator"""

    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TESTTYPE = "basic",
        seed: int | None = None,
        mu: float = 0,
    ) -> None:
        """initiation

        Args:
            test_info (TestInfo): significant test parameters
            generators (dict[str, DistGenerator]): distribution generators
            iters (int, optional): simulation iteration. Defaults to 10000.
            test_type (TESTTYPE, optional): significant test operation. Defaults to "basic".
            seed (Optional[int], optional): fixed seed number. Defaults to None.
            mu (float, optional): comparetion value for one-sample T test.
                Defaults to 0.

        Raises:
            ValueError: undifined method
        """
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
        """update sample"""
        self.test_param.update(a=self.generators["X"].create_sample())
        if self.test_info["method"] != "one-sample":
            self.test_param.update(b=self.generators["Y"].create_sample())


class WilcoxonTestSimulator(StatTestSimulator):
    """wilcoxon or Mann-Whitney U test simulator"""

    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TESTTYPE = "basic",
        seed: int | None = None,
    ) -> None:
        """initiation

        Args:
            test_info (TestInfo): significant test parameters
            generators (dict[str, DistGenerator]): distribution generators
            iters (int, optional): simulation iteration. Defaults to 10000.
            test_type (TESTTYPE, optional): significant test operation. Defaults to "basic".
            seed (Optional[int], optional): fixed seed number. Defaults to None.

        Raises:
            ValueError: undifined method
        """
        super().__init__(test_info, generators, iters, test_type, seed)

        if self.test_info["method"] == "normal":
            self.test_funcs = stats.mannwhitneyu
        elif self.test_info["method"] == "paired":
            self.test_funcs = stats.wilcoxon
        else:
            raise ValueError("t test method must be [normal, paired]")

    def sample_update(self) -> None:
        """update sample"""
        self.test_param.update(x=self.generators["X"].create_sample())
        self.test_param.update(y=self.generators["Y"].create_sample())


class BrunnerMunzelTestSimulator(StatTestSimulator):
    """BrunnerMunzel test simulator"""

    def __init__(
        self,
        test_info: TestInfo,
        generators: dict[str, DistGenerator],
        iters: int = 10000,
        test_type: TESTTYPE = "basic",
        seed: int | None = None,
    ) -> None:
        """initiation

        Args:
            test_info (TestInfo): significant test parameters
            generators (dict[str, DistGenerator]): distribution generators
            iters (int, optional): simulation iteration. Defaults to 10000.
            test_type (TESTTYPE, optional): significant test operation. Defaults to "basic".
            seed (Optional[int], optional): fixed seed number. Defaults to None.

        Raises:
            ValueError: undifined method
        """
        super().__init__(test_info, generators, iters, test_type, seed)

        self.test_funcs = stats.brunnermunzel

    def sample_update(self) -> None:
        """update sample"""
        self.test_param.update(x=self.generators["X"].create_sample())
        self.test_param.update(y=self.generators["Y"].create_sample())


def build_simulator(name: str, **args) -> StatTestSimulator:
    """build simulator

    Args:
        name (str): significant test name

    Returns:
        StatTestSimulator: significant test simulator
    """
    if name == "t_test":
        return TTestSimulator(**args)
    if name == "wilcoxon_test":
        return WilcoxonTestSimulator(**args)
    if name == "brunner_munzel_test":
        return BrunnerMunzelTestSimulator(**args)
    return StatTestSimulator(**args)
