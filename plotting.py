import os.path
import pandas as pd
import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt
from datetime import datetime


class GraphPlotter:
    """
    The graphics plotter class which visualises calculated metrics
    """
    METRICS_PATH = 'metrics'
    PROBABILITIES_TICKS_CNT = 10

    def __init__(self, dists: pd.DataFrame, prop_loss_region: int, packages_region: int):
        self.dists = dists
        self.prop_loss_region = prop_loss_region
        self.packages_region = packages_region
        self.prop_loss_dict = {}
        self.signal_reception_dict = {}
        self.init_metrics()

    def generate_distance_regions(self, region_size: int) -> List[int]:
        """
        Split all the vehicle distances into regions for further visualisation
        :param region_size: The size of distance region
        :return: List of distance regions
        """
        dists_min = self.dists.min(skipna=True, axis=None, numeric_only=True)
        dists_max = self.dists.max(skipna=True, axis=None, numeric_only=True)
        min_dist = int(np.floor(dists_min / region_size) * region_size)
        max_dist = int(np.floor(dists_max / region_size) * region_size)
        dist_regions = [i for i in range(min_dist, max_dist + region_size, region_size)]
        return dist_regions

    def init_metrics(self):
        """
        Initialise metrics containers
        """
        prop_loss_labels = self.generate_distance_regions(self.prop_loss_region)
        packages_labels = self.generate_distance_regions(self.packages_region)
        self.prop_loss_dict = {label: {'prop_loss': 0, 'v_num': 0} for label in prop_loss_labels}
        self.signal_reception_dict = {label: {'correct': 0, 'total': 0} for label in packages_labels}

    def process_row(self, dist_row: pd.Series, prop_loss_df: pd.DataFrame, signal_reception_df: pd.DataFrame,
                    timestamp: float) -> None:
        """
        Match plot data with its distance in a single row
        :param dist_row: Row of a dists dataframe
        :param prop_loss_df: Dataframe containing propagation loss of signal
        :param signal_reception_df: Dataframe containing signal reception
        :param timestamp: The timestamp of a given row
        """
        v0 = dist_row.name
        for v1, dist in dist_row.items():
            if not np.isnan(dist):
                prop_loss_val = prop_loss_df[v1][timestamp, v0]
                signal_reception_val = signal_reception_df[v1][timestamp, v0]
                prop_loss_dist = int(np.floor(dist / self.prop_loss_region) * self.prop_loss_region)
                packages_dist = int(np.floor(dist / self.packages_region) * self.packages_region)
                self.prop_loss_dict[prop_loss_dist]['prop_loss'] += prop_loss_val
                self.prop_loss_dict[prop_loss_dist]['v_num'] += 1
                self.signal_reception_dict[packages_dist]['correct'] += signal_reception_val
                self.signal_reception_dict[packages_dist]['total'] += 1

    def generate_distance_data_dependencies(self, prop_loss: pd.DataFrame, signal_reception: pd.DataFrame) -> None:
        """
        Match plot data with its distance
        """
        for timestamp, new_df in self.dists.groupby(level=0):
            new_df = new_df.droplevel(0)
            new_df.apply(self.process_row, axis=1, args=(prop_loss, signal_reception, timestamp))

    def create_prop_loss_plots(self) -> Tuple[List[int], List[float]]:
        """
        Plot propagation loss values averaged by distance
        """
        plot_x, plot_y = [], []
        for key in self.prop_loss_dict:
            if self.prop_loss_dict[key]:
                plot_x.append(key)
                prop_loss_sum = self.prop_loss_dict[key]['prop_loss']
                v_num = self.prop_loss_dict[key]['v_num']
                plot_y.append(prop_loss_sum / v_num)
        return plot_x, plot_y

    def create_prr_plots(self) -> Tuple[List[int], List[float]]:
        """
        Plot packet reception ratio values averaged by distance
        """
        plot_x, plot_y = [], []
        for key in self.signal_reception_dict:
            if self.signal_reception_dict[key]:
                plot_x.append(key)
                correct = np.sum(self.signal_reception_dict[key]['correct'])
                total = np.sum(self.signal_reception_dict[key]['total'])
                plot_y.append(correct / total)
        return plot_x, plot_y

    def create_plr_plots(self) -> Tuple[List[int], List[float]]:
        """
        Plot propagation loss ratio values averaged by distance
        """
        plot_x, plot_y = [], []
        for key in self.signal_reception_dict:
            if self.signal_reception_dict[key]:
                plot_x.append(key)
                correct = np.sum(self.signal_reception_dict[key]['correct'])
                total = np.sum(self.signal_reception_dict[key]['total'])
                incorrect = total - correct
                plot_y.append(incorrect / total)
        return plot_x, plot_y

    def plot_prop_loss(self, ax: plt.Axes, fig: plt.Figure, dir_path: str):
        x_plots, y_plots = self.create_prop_loss_plots()
        ax.set_title('PL (propagation loss)')
        ax.plot(x_plots, y_plots)
        ax.set_ylabel('PL', labelpad=12, rotation='horizontal')
        fig.savefig(fname=f'{dir_path}/PL.eps', format='eps')

    def plot_prr(self, ax: plt.Axes, probabilities_ticks: List[float], fig: plt.Figure, dir_path: str):
        x_plots, y_plots = self.create_prr_plots()
        ax.set_title('PRR (packet reception ratio)')
        ax.plot(x_plots, y_plots)
        ax.set_yticks(probabilities_ticks, probabilities_ticks)
        ax.set_ylabel('PRR', labelpad=12, rotation='horizontal')
        fig.savefig(fname=f'{dir_path}/PRR.eps', format='eps')

    def plot_plr(self, ax: plt.Axes, probabilities_ticks: List[float], fig: plt.Figure, dir_path: str):
        x_plots, y_plots = self.create_plr_plots()
        ax.set_title('PLR (packet loss ratio)')
        ax.plot(x_plots, y_plots)
        ax.set_yticks(probabilities_ticks, probabilities_ticks)
        ax.set_ylabel('PLR', labelpad=12, rotation='horizontal')
        fig.savefig(fname=f'{dir_path}/PLR.eps', format='eps')

    def plot_metrics(self) -> None:
        """
        Plot the simulation metrics using matplotlib and save it
        """
        prop_loss_fig, prop_loss_ax = plt.subplots()
        prr_fig, prr_ax = plt.subplots()
        plr_fig, plr_ax = plt.subplots()
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        axes = [prop_loss_ax, prr_ax, plr_ax]
        for el in axes:
            el.grid()
            el.margins(x=0, y=0)
            el.set_xlabel('Distance (m)')
        prop_loss_ax, prr_ax, plr_ax = axes[0], axes[1], axes[2]
        probabilities_ticks = [i / self.PROBABILITIES_TICKS_CNT for i in range(0, self.PROBABILITIES_TICKS_CNT + 1)]
        if not os.path.exists(self.METRICS_PATH):
            os.mkdir(self.METRICS_PATH)
        dir_path = self.METRICS_PATH + '/' + datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
        os.mkdir(dir_path)
        self.plot_prop_loss(prop_loss_ax, prop_loss_fig, dir_path)
        self.plot_prr(prr_ax, probabilities_ticks, prr_fig, dir_path)
        self.plot_plr(plr_ax, probabilities_ticks, plr_fig, dir_path)
