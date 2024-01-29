# !/usr/bin/env python3

import platform
from typing import Optional

import pkg_resources
import streamlit as st

from src.simulate import generate, simulate, visualize
from src.utils import logging


class AppBuilder:
    def __init__(self) -> None:
        self.logger = logging.set_logger("warning")

        self.dist_name: list[str] = ["norm", "lognorm", "uniform"]
        self.dist_param: dict[str, dict[str, float | int]] = {"X": {}, "Y": {}}
        self.test_name: list[str] = ["t_test", "wilcoxon_test"]

        self.generators: dict[str, generate.DistGenerator] = {
            "X": generate.build_generator(self.dist_name[0], **self.dist_param["X"]),
            "Y": generate.build_generator(self.dist_name[0], **self.dist_param["Y"]),
        }
        self.simulator: simulate.StatTestSimulator = simulate.build_simulator(
            self.test_name[0],
            test_info={
                "method": "welch",
                "alternative": "two-sided",
                "alpha": 0.05,
            },
            generators=self.generators,
            iters=10000,
        )
        self.visualizer: visualize.Visualizer = visualize.Visualizer(self.simulator)

        self.simulation_flag: bool = False

    def __call__(self) -> None:
        st.set_page_config(
            page_title="統計的仮説検定シミュレーター",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        self.sidebar_component()
        self.body_components()

    def sidebar_component(self) -> None:
        st.sidebar.subheader("親ページ")
        st.sidebar.link_button(
            "BiostatMoon",
            "https://yasuih777.github.io/biostatmoon/qmds/",
            type="primary",
        )

        st.sidebar.subheader("その他リンク")
        st_col = st.sidebar.columns(2)
        with st_col[0]:
            st.link_button("X(Twitter)", "https://twitter.com/yugetsubiostat")
        with st_col[1]:
            st.link_button(
                "Github", "https://github.com/yasuih777/simulation_significant_test"
            )

        st.sidebar.subheader("使い方")
        st.sidebar.markdown(
            "1. 確率分布を設定します\n"
            "2. 検定の方法を設定します\n"
            "3. 「シミュレーション開始」をクリックします\n"
            "4. P値が一様になっていることや検出力をチェックします"
        )

        st.sidebar.subheader("Licence")
        st.sidebar.link_button("MIT Licence", "https://opensource.org/license/mit/")

        with st.sidebar.expander("Python environment"):
            st.text(f"Python version: {platform.python_version()}")
            st.text(f"Python pkgs:")
            for pkg in pkg_resources.working_set:
                st.text(pkg)

    def body_components(self) -> None:
        st.title("統計的仮説検定シミュレーター")

        st.header("1. サンプルが従う確率分布の設定")
        gparam_body = st.columns(2)
        with gparam_body[0]:
            with st.container(border=True):
                self.__generate_input("X")
        with gparam_body[1]:
            with st.container(border=True):
                self.__generate_input("Y")

        fig, ax = visualize.create_figure(figsize=(8, 3))
        self.visualizer.generate_density(ax)
        st.pyplot(fig)
        fig.clear()

        st.header("2. 統計的仮説検定の設定")
        with st.container(border=True):
            tparam_body = st.columns(3)
            with tparam_body[0]:
                test_name, method = self.__test_input()
            with tparam_body[1]:
                alternative = st.selectbox(
                    "対立仮説を棄却する向き", ["two-sided", "greater", "less"]
                )
                alpha = st.number_input(
                    "有意水準", min_value=0.0, max_value=1.0, value=0.05, step=0.01
                )
            with tparam_body[2]:
                iters = st.number_input(
                    "シミュレーション回数", min_value=1000, value=10000, step=100
                )
                test_type = st.selectbox("検定のフロー", ["basic"])

        test_info = {
            "method": method,
            "alternative": alternative,
            "alpha": alpha,
        }
        self.simulator = simulate.build_simulator(
            test_name,
            test_info=test_info,
            generators=self.generators,
            iters=iters,
            test_type=test_type,
        )

        self.__test_discription()

        self.simulation_flag = st.button("シミュレーション開始", type="primary")
        if self.simulation_flag:
            self.__simulation()

            with st.container(border=True):
                st.text("有意差が認められた割合")
                st.write(f":red[{self.simulator.per_signicant() * 100:.2f}%]")

            fig, axes = visualize.create_figure(
                nrows=1, ncols=1, figsize=(8, 3), tight_layout=True
            )
            self.visualizer.p_histogram(axes)
            st.pyplot(fig)

    def __generate_input(self, name: str) -> None:
        dist_name = st.selectbox(f"{name}群の分布", self.dist_name)
        self.dist_param[name].update(
            sample_size=st.number_input(f"{name}群のサンプルサイズ", min_value=1, value=50)
        )

        if dist_name == "norm":
            self.dist_param[name].update(
                {"mu": st.number_input(f"{name}: mu", value=0.0)}
            )
            self.dist_param[name].update(
                {"sigma": st.number_input(f"{name}: sigma", min_value=0.0, value=1.0)}
            )
        elif dist_name == "lognorm":
            self.dist_param[name].update(
                {"mu": st.number_input(f"{name}: mu", min_value=-0.0, value=1.0)}
            )
            self.dist_param[name].update(
                {"sigma": st.number_input(f"{name}: sigma", min_value=0.0, value=1.0)}
            )
        elif dist_name == "uniform":
            self.dist_param[name].update(
                {"a": st.number_input(f"{name}: a", value=0.0)}
            )
            self.dist_param[name].update(
                {"b": st.number_input(f"{name}: b", value=1.0)}
            )

        self.generators[name] = generate.build_generator(
            dist_name, **self.dist_param[name]
        )

        del dist_name

    def __test_input(self) -> tuple[str, Optional[str]]:
        method: Optional[str]
        test_name = st.selectbox("検定", self.test_name)

        if test_name == "t_test":
            method = st.selectbox("T検定のメソッド", ["welch", "student", "paired"])
        elif test_name == "wilcoxon_test":
            method = st.selectbox("Wilcoxon(or MannwhitneyのU)検定のメソッド", ["normal", "paired"])
        else:
            method = None

        return test_name, method

    def __test_discription(self) -> None:
        simulator = self.simulator
        generators = self.simulator.generators

        if simulator.test_info["alternative"] == "two-sided":
            alternative = "両側"
        elif simulator.test_info["alternative"] == "greater":
            alternative = "片側(>)"
        elif simulator.test_info["alternative"] == "less":
            alternative = "片側(<)"

        if isinstance(simulator, simulate.TTestSimulator):
            if simulator.test_info["method"] == "welch":
                test_name = "WelchのT検定"
            elif simulator.test_info["method"] == "student":
                test_name = "StudentのT検定"
            elif simulator.test_info["method"] == "paired":
                test_name = "対応のあるのT検定"
            elif simulator.test_info["method"] == "one-sample":
                test_name = "1標本T検定"
        elif isinstance(simulator, simulate.WilcoxonTestSimulator):
            if simulator.test_info["method"] == "normal":
                test_name = "MannwhitneyのU検定"
            elif simulator.test_info["method"] == "paired":
                test_name = "Wilcoxonの符号付き順位検定"

        st.text(f"以下の設定でシミュレーションを{simulator.iters}回行う")
        st.text(
            f'X群: {generators["X"].dist_name}に従う'
            f'サンプルサイズ{generators["X"].sample_size}の標本'
        )
        if simulator.test_info["method"] != "one-sample":
            st.text(
                f'Y群: {generators["Y"].dist_name}に従う'
                f'サンプルサイズ{generators["Y"].sample_size}の標本'
            )

        st.text(
            f"検定: {alternative}での{test_name}、"
            f'有意水準は{simulator.test_info["alpha"] * 100:.2f}%とする'
        )

        if simulator.test_type == "basic":
            pass

    def __simulation(self) -> None:
        self.simulator.execute()
        self.visualizer.reset_simulator(self.simulator)
