# !/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from src.simulate import simulate


class Visualizer:
    def __init__(self, simulator: simulate.StatTestSimulator) -> None:
        self.simulator = simulator

    def reset_simulator(self, simulator: simulate.StatTestSimulator) -> None:
        self.simulator = simulator

    def generate_density(self, axes: plt.Axes) -> plt.Axes:
        points_x = self.simulator.generators["X"].density_points(1000)
        points_y = self.simulator.generators["Y"].density_points(1000)
        label_x = self.simulator.generators["X"].dist_name
        label_y = self.simulator.generators["Y"].dist_name

        axes.fill_between(
            points_x["x"], points_x["y"], alpha=0.5, label=f"X: {label_x}"
        )
        axes.fill_between(
            points_y["x"], points_y["y"], alpha=0.5, label=f"Y: {label_y}"
        )
        axes.set_title("Generator probability distribution")
        axes.set_xlabel("X")
        axes.set_ylabel("Density")
        axes.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        return axes

    def p_histogram(self, axes: plt.Axes) -> plt.Axes:
        axes.hist(
            self.simulator.results["p_values"],
            bins=50,
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

    def stat_density(self, axes: plt.Axes) -> plt.Axes:
        points = self.simulator.density_points()

        axes.hist(
            self.simulator.results["stat_values"],
            bins=int(np.sqrt(self.simulator.iters)),
            density=True,
            color="grey",
            ec="black",
            alpha=0.8,
        )
        axes.plot(
            points["h0_x"], points["h0_y"], color="black", linestyle="--", label="H0"
        )
        print(points["h0_y"], points["h1_y"])
        axes.plot(points["h1_x"], points["h1_y"], color="black", label="H1")
        axes.fill_between(
            points["h1_upperx"], points["h1_uppery"], color="blue", alpha=0.4
        )
        axes.fill_between(
            points["h1_lowerx"], points["h1_lowery"], color="blue", alpha=0.4
        )
        axes.axvline(self.simulator.rejection[0], color="blue", linestyle="--")
        axes.axvline(self.simulator.rejection[1], color="blue", linestyle="--")
        axes.set_title("Statistic value distribution by simulator")
        axes.set_xlabel("Statistics value")
        axes.set_ylabel("Density")
        axes.legend()

        return axes


def create_figure(**args) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(**args)

    return fig, axes
