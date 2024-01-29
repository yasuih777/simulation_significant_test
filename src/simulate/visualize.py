# !/usr/bin/env python3

import matplotlib.pyplot as plt

from src.simulate import simulate, generate


class Visualizer:
    def __init__(self, generators: dict[str, generate.DistGenerator]) -> None:
        self.generators = generators
        self.simulator: simulate.StatTestSimulator

    def update_simulator(self, simulator: simulate.StatTestSimulator) -> None:
        self.simulator = simulator

    def generate_density(self, axes: plt.Axes) -> plt.Axes:
        for key, generator in self.generators.items():
            points = generator.density_points(1000)
            label = generator.dist_name

            axes.fill_between(
                points["x"], points["y"], alpha=0.5, label=f"{key}: {label}"
            )
        axes.set_title("Generator probability distribution")
        axes.set_xlabel("X")
        axes.set_ylabel("Density")
        axes.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        return axes

    def p_histogram(self, axes: plt.Axes) -> plt.Axes:
        bins = 100
        axes.axhline(y = self.simulator.iters / bins, linestyle="--", color="black", alpha=0.5)
        axes.hist(
            self.simulator.p_values,
            bins=bins,
            range=(0, 1),
            color="grey",
            ec="black",
            alpha=0.8,
        )
        axes.axvspan(0, self.simulator.test_info["alpha"], color="red", alpha=0.4)
        axes.set_title("P value histogram by simulator")
        axes.set_xlabel("P value")
        axes.set_ylabel("Count")

        return axes


def create_figure(**args) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(**args)

    return fig, axes
