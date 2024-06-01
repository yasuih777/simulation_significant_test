# !/usr/bin/env python3

from typing import Literal

import numpy as np
from scipy import stats

from src.utils.typing import GeneratorFunc

"""
Basic Generator
"""


class DistGenerator:
    """distribution generator

    Attributes:
        sample_size (int): sample size
        funcs (GeneratorFunc): functions for generator
        func_param (dict[str, int | float | np.ndarray]): functions parameters
        plot_prob (float): plot x axis range of distribution function
        dist_name (str): distribution name
        dist_type (Literal["discrete", "continue"]): distribution type
    """

    def __init__(self, sample_size: int = 50) -> None:
        """intiation

        Args:
            sample_size (int, optional): sample size. Defaults to 50.
        """
        self.sample_size = sample_size

        self.funcs: GeneratorFunc
        self.func_param: dict[str, int | float | np.ndarray]
        self.plot_prob: float = 0.999

        self.dist_name: str
        self.dist_type: Literal["discrete", "continue"]

    def create_sample(self) -> np.ndarray:
        """create sample by distribution

        Returns:
            np.ndarray: sample
        """
        param = self.func_param.copy()
        param.update(size=self.sample_size)

        return self.funcs["rand"](**param)

    def plot_range(self) -> tuple[float, float]:
        """return plot range

        Raises:
            NotImplementedError: must override

        Returns:
            tuple[float, float]: x axis range for graph
        """
        raise NotImplementedError("Must override!!")

    def density_points(self, delta: int = 100) -> dict[str, np.ndarray]:
        """return density points

        Args:
            delta (int, optional): points size. Default is 100.
        """
        ranges = self.plot_range()

        x_vec = np.linspace(ranges[0], ranges[1], delta)
        param = self.func_param.copy()
        param.update(x=x_vec)
        y_vec = self.funcs["stat_prob"](**param)

        return {"x": x_vec, "y": y_vec}


"""
Continue distribution Generator
"""


class NormGenerator(DistGenerator):
    """norm distribution generator"""

    def __init__(self, sample_size: int = 50, mu: float = 0, sigma: float = 1) -> None:
        """initiation

        Args:
            sample_size (int, optional): sample size. Defaults to 50.
            mu (float, optional): mean of population distribution. Defaults to 0.
            sigma (float, optional): standard deviation of population distribution.
                Defaults to 1.
        """
        super().__init__(sample_size)

        self.funcs = {
            "rand": stats.norm.rvs,
            "stat_prob": stats.norm.pdf,
        }
        self.func_param = {"loc": mu, "scale": sigma}

        self.dist_name = f"Norm({mu}, {sigma})"
        self.dist_type = "continue"

    def plot_range(self) -> tuple[float, float]:
        """return plot range

        Returns:
            tuple[float, float]: x axis range for graph
        """
        param = self.func_param.copy()
        param.update(confidence=self.plot_prob)

        return stats.norm.interval(**param)


class LogNormGenerator(DistGenerator):
    """log-norm distribution generator"""

    def __init__(self, sample_size: int = 50, mu: float = 0, sigma: float = 1) -> None:
        """initiation

        Args:
            sample_size (int, optional): sample size. Defaults to 50.
            mu (float, optional): mean of population distribution. Defaults to 0.
            sigma (float, optional): standard deviation of population distribution.
                Defaults to 1.
        """
        super().__init__(sample_size)
        self.plot_prob = 0.95

        self.funcs = {
            "rand": stats.lognorm.rvs,
            "stat_prob": stats.lognorm.pdf,
        }
        self.func_param = {"scale": np.exp(mu), "s": sigma}

        self.dist_name = f"LogNorm({mu}, {sigma})"
        self.dist_type = "continue"

    def plot_range(self) -> tuple[float, float]:
        """return plot range

        Returns:
            tuple[float, float]: x axis range for graph
        """
        param = self.func_param.copy()
        param.update(q=self.plot_prob)

        return (0, stats.lognorm.ppf(**param))


class UniGenerator(DistGenerator):
    """uniform distribution generator"""

    def __init__(self, sample_size: int = 50, a: float = 0, b: float = 1) -> None:
        """initiation

        Args:
            sample_size (int, optional): sample size. Defaults to 50.
            a (float, optional): minimum of population distribution. Defaults to 0.
            b (float, optional): maximum of population distribution. Defaults to 1.
        """
        super().__init__(sample_size)

        self.funcs = {
            "rand": stats.uniform.rvs,
            "stat_prob": stats.uniform.pdf,
        }
        self.func_param = {"loc": a, "scale": b - a}

        self.dist_name = f"Uni({a}, {b})"
        self.dist_type = "continue"

    def plot_range(self) -> tuple[float, float]:
        """return plot range

        Returns:
            tuple[float, float]: x axis range for graph
        """
        param = self.func_param.copy()

        return (param["loc"], param["loc"] + param["scale"])  # type: ignore


class GammaGenerator(DistGenerator):
    """gamma distribution generator"""

    def __init__(
        self, sample_size: int = 50, alpha: float = 1, beta: float = 1
    ) -> None:
        """initiation

        Args:
            sample_size (int, optional): sample size. Defaults to 50.
            alpha (float, optional): parameter of population distribution. Defaults to 0.
            beta (float, optional): parameter of population distribution. Defaults to 1.
        """
        super().__init__(sample_size)

        self.funcs = {
            "rand": stats.gamma.rvs,
            "stat_prob": stats.gamma.pdf,
        }
        self.func_param = {"a": alpha, "scale": beta}

        self.dist_name = f"Gamma({alpha}, {beta})"
        self.dist_type = "continue"

    def plot_range(self) -> tuple[float, float]:
        """return plot range

        Returns:
            tuple[float, float]: x axis range for graph
        """
        param = self.func_param.copy()
        param.update(q=self.plot_prob)

        return (0, stats.gamma.ppf(**param))


def build_generator(name: str, **args) -> DistGenerator:
    """build generator

    Args:
        name (str): distribution name

    Returns:
        DistGenerator: distriution generator
    """
    if name == "norm":
        return NormGenerator(**args)
    if name == "lognorm":
        return LogNormGenerator(**args)
    if name == "gamma":
        return GammaGenerator(**args)
    if name == "uniform":
        return UniGenerator(**args)
    return DistGenerator(**args)
