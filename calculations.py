import pandas as pd
import numpy as np
from typing import Optional
from scipy.spatial.distance import pdist, squareform


def calc_highway_los_nlosv(dists: pd.DataFrame, lognorm_mean: int, lognorm_sigma: int,
                           signal_freq: float) -> pd.DataFrame:
    """
    Calculate propagation loss for LOS and NLOSv blockage for highway scenario
    :param lognorm_mean: Mean of lognormal distribution
    :param lognorm_sigma: Sigma of lognormal distribution
    :param signal_freq: Frequency of signal
    :param dists: Distances between vehicles.
    :return: Dataframe containing calculated propagation loss
    """
    lognorm = np.random.lognormal(lognorm_mean, lognorm_sigma, size=dists.shape)
    signal_freq = np.log10(np.full(dists.shape, signal_freq))
    prop_loss = 32.4 + 20 * (np.log10(dists) + np.log10(signal_freq)) + lognorm
    return prop_loss


class CalcProc:
    """
    The calculation processor class which enacts most of the data processing
    """
    LOGNORM_MEAN = 0
    LOGNORM_SIGMA = 3
    PROP_LOSS_FUNCTIONS = {
        'highway': calc_highway_los_nlosv
    }

    def __init__(self, sim_data: pd.DataFrame, env_scenario: str, signal_freq: float, signal_power: float,
                 reception_threshold: float):
        self.cords = sim_data
        self.env_scenario = env_scenario
        self.signal_freq = signal_freq
        self.signal_power = signal_power
        self.reception_threshold = reception_threshold
        self.preprocess_cords()
        self.dists = self.calc_dists()
        self.prop_loss = self.calc_prop_loss()
        self.reception_rate = self.calc_signal_reception()

    def preprocess_cords(self) -> None:
        """
        Preprocess vehicle coordinates
        """
        self.cords.rename(columns={'timestep_time': 'timestamp'}, inplace=True)
        self.cords = self.cords.dropna().sort_values(['timestamp', 'vehicle_id'])
        self.cords['vehicle_id'] = self.cords['vehicle_id'].str.replace('.', '_')
        self.cords.set_index(['timestamp', 'vehicle_id'], inplace=True)

    def calc_dists(self) -> pd.DataFrame:
        """
        Calculate distances between vehicles based on their position
        :return: Dataframe containing calculated distances in a format [timestamp, v0][v1]
        """
        def transform(df: pd.DataFrame) -> Optional[pd.DataFrame]:
            if df.shape[0] > 1:
                dists = squareform(pdist(list(zip(df['vehicle_x'], df['vehicle_y']))))
                new_df = pd.DataFrame(dists, index=df.index.get_level_values(1), columns=df.index.get_level_values(1))
                return new_df

        dists = self.cords.groupby(level=0).apply(transform)
        dists.mask(np.isclose(dists, 0), inplace=True)
        return dists

    def calc_prop_loss(self) -> Optional[pd.DataFrame]:
        """
        Calculate propagation loss
        :return: Dataframe containing calculated propagation loss
        """
        ans_df = self.PROP_LOSS_FUNCTIONS[self.env_scenario](dists=self.dists, lognorm_mean=self.LOGNORM_MEAN,
                                                             lognorm_sigma=self.LOGNORM_SIGMA,
                                                             signal_freq=self.signal_freq)
        return ans_df

    def calc_signal_reception(self) -> pd.DataFrame:
        """
        Calculate vehicles signal reception rate
        :return: Dataframe containing calculated signal reception
        """
        starting_power = np.full(self.prop_loss.shape, self.signal_power)
        reception_rate = starting_power - self.prop_loss >= self.reception_threshold
        return reception_rate
