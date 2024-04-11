import pandas as pd
import numpy as np
import os
import re
from typing import Dict, Any, Tuple, Optional, List
from copy import copy
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt


class UserInput:
    """
    The user input class which is used for initial data retrieval and processing
    """
    VALID_PARAMETERS = {
        'dataFilepath': str,
        'csvSep': str,
        'signalFreq': int,
        'receptionThreshold': float,
        'signalPower': int,
        'envScenario': str,
        'propLossRegion': int,
        'packagesRegion': int
    }
    DEFAULT_PARAMETERS = {
        'csvSep': ';',
        'envScenario': 'highway',
        'propLossRegion': 500,
        'packagesRegion': 500
    }
    NECESSARY_PARAMETERS = {'dataFilepath', 'signalFreq', 'receptionThreshold', 'signalPower'}
    PANDAS_READERS = {
        '.csv': pd.read_csv,
        '.xls': pd.read_excel,
        '.xlsx': pd.read_excel
    }
    VALID_DATA_COLUMNS = {
        'timestep_time': float,
        'vehicle_angle': float,
        'vehicle_id': str,
        'vehicle_x': float,
        'vehicle_y': float
    }
    VALID_SCENARIOS = {'highway', 'urban'}

    def __init__(self):
        pass

    def check_config_parameters(self, dct: Dict[str, Any]) -> bool:
        """
        Check if provided config parameters are correct. Should be called after parsing the config file
        :param dct: Config parameters
        """
        ans = True
        for param in dct:
            if param not in self.DEFAULT_PARAMETERS:
                if self.VALID_PARAMETERS[param] in [int, float]:
                    try:
                        dct[param] = self.VALID_PARAMETERS[param](dct[param])
                    except ValueError:
                        print(f'Некорректный тип параметра {param}, требуется {self.VALID_PARAMETERS[param]}')
                        ans = False
        if not os.path.isfile(dct['dataFilepath']):
            print('Введён некорректный путь к файлу данных симуляции')
            ans = False
        if dct['envScenario'] not in self.VALID_SCENARIOS:
            print(
                f'Введён некорректный сценарий симуляции {dct["envScenario"]}, доступные опции: {self.VALID_SCENARIOS}')
            ans = False
        return ans

    def read_config_file(self, filename: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the config file is not corrupted and parse it into a dict
        :param filename: Path to configuration file
        :return: Boolean value - was the parsing successful, parsed dict
        """
        with open(filename) as f:
            lines = f.readlines()
        config_params = copy(self.DEFAULT_PARAMETERS)
        reg_check = '[^ =]*=[^ =]*'
        correct_params = True
        for i in range(len(lines)):
            line = lines[i].strip()
            if not re.fullmatch(reg_check, line):
                print(f'Некорректный формат файла конфигурации, ошибка в строке {i + 1}', line, sep='\n')
                correct_params = False
            else:
                param, val = line.split('=')
                if param not in self.VALID_PARAMETERS:
                    correct_params = False
                    print(f'Некорректный параметр симуляции, ошибка в строке {i + 1}', line, sep='\n')
                else:
                    if param in config_params and param not in self.DEFAULT_PARAMETERS:
                        print(f'Повторяющийся параметр симуляции {param} в строке {i + 1}')
                        correct_params = False
                    else:
                        config_params[param] = val
        missing_params = self.NECESSARY_PARAMETERS - config_params.keys()
        if missing_params:
            print(f'В конфигурационном файле отсутствуют обязательные параметры симуляции: {missing_params}')
            correct_params = False
        else:
            correct_params = correct_params and self.check_config_parameters(config_params)
        return correct_params, config_params

    @staticmethod
    def check_duplicate_columns(filename: str, ext: str, configs: Dict[str, Any]) -> bool:
        """
        Check if a simulation data file contains duplicate columns
        :param filename: Path to the simulation data file
        :param ext: The extension of the simulation data file
        :param configs: Configuration data
        :return: Does the data file contain duplicate columns
        """
        if ext == '.csv':
            first_row = pd.read_csv(filename, nrows=1, sep=configs['csvSep'])
        else:
            first_row = pd.read_excel(filename, nrows=1)
        columns = list(first_row.columns)
        for i in range(len(columns)):
            col_name1 = columns[i]
            for j in range(i + 1, len(columns)):
                col_name2 = columns[j]
                if col_name2.startswith(col_name1):
                    return True
        return False

    def read_sim_datafile(self, filename: str, configs: Dict[str, Any]) -> Tuple[bool, pd.DataFrame]:
        """
        Read a simulation data file and parse it to a pandas dataframe
        :param filename: Path to the simulation data file
        :param configs: Configuration data
        :return: Boolean value - was the parsing successful, parsed pandas dataframe
        """
        ans = True
        df = None
        ext = os.path.splitext(filename)[1].lower()
        if ext in self.PANDAS_READERS:
            # noinspection PyTypedDict
            reader = self.PANDAS_READERS[ext]
            try:
                if self.check_duplicate_columns(filename, ext, configs):  # Checking for duplicate column labels
                    print("Обнаружены дубликаты в названиях столбцов в файле данных симуляции")
                    ans = False
                else:
                    if ext == '.csv':
                        df = reader(filename, sep=configs['csvSep'], dtype=self.VALID_DATA_COLUMNS,
                                    usecols=self.VALID_DATA_COLUMNS)
                    else:
                        df = reader(filename, dtype=self.VALID_DATA_COLUMNS, usecols=self.VALID_DATA_COLUMNS)
            except Exception as exc:
                print('Некорректный формат файла данных симуляции')
                print(exc)
                ans = False
        else:
            print('Неподдерживаемое расширение файла данных симуляции, cписок поддерживаемых расширений:')
            print(self.PANDAS_READERS.keys())
            ans = False
        return ans, df

    def start(self) -> Optional[Tuple[Dict[str, Any], pd.DataFrame]]:
        """
        Start the user dialogue
        :return: Parsed configuration data, simulation data
        """
        print('Введите путь к файлу конфигурации: ', end='')
        conf = input()
        while not os.path.isfile(conf):
            print('Введён некорректный путь к файлу конфигурации, повторите ввод: ', end='')
            conf = input()
        correct_params, config_params = self.read_config_file(conf)
        if correct_params:
            sim_data = config_params['dataFilepath']
            correct_data, sim_data = self.read_sim_datafile(sim_data, config_params)
            if correct_data:
                return config_params, sim_data


class CalcProc:
    """
    The calculation processor class which enacts most of the data processing
    """
    LOGNORM_MEAN = 0
    LOGNORM_SIGMA = 3

    def __init__(self, conf: Dict[str, Any], sim_data: pd.DataFrame):
        self.cords = sim_data
        self.conf = conf
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

    def calc_highway_los_nlosv(self, dists: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate propagation loss for LOS and NLOSv blockage for highway scenario
        :param dists: Distances between vehicles.
        :return: Dataframe containing calculated propagation loss
        """
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.log10(np.full(dists.shape, self.conf['signalFreq']))
        prop_loss = 32.4 + 20 * (np.log10(dists) + np.log10(signal_freq)) + lognorm
        return prop_loss

    def calc_urban_los_nlosv(self, dists: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate propagation loss for LOS and NLOSv blockage for urban scenario
        :param dists: Distances between vehicles.
        :return: Dataframe containing calculated propagation loss
        """
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.full(dists.shape, self.conf['signalFreq'])
        prop_loss = 38.77 + 16.7 * np.log10(dists) + 18.2 * np.log10(signal_freq) + lognorm
        return prop_loss

    def calc_urban_nlos(self, dists: pd.DataFrame) -> pd.DataFrame:
        #  TODO: change lognorm to 4
        """
        Calculate propagation loss for NLOS blockage for urban scenario
        :param dists: Distances between vehicles.
        :return: Dataframe containing calculated propagation loss
        """
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.full(dists.shape, self.conf['signalFreq'])
        prop_loss = 36.85 + 30 * np.log10(dists) + 18.9 * np.log10(signal_freq) + lognorm
        return prop_loss

    def calc_prop_loss(self) -> Optional[pd.DataFrame]:
        """
        Calculate propagation loss
        :return: Dataframe containing calculated propagation loss
        """
        ans_df = None
        if self.conf['envScenario'] == 'highway':
            ans_df = self.calc_highway_los_nlosv(self.dists)
        return ans_df

    def calc_signal_reception(self) -> pd.DataFrame:
        starting_power = np.full(self.prop_loss.shape, self.conf['signalPower'])
        reception_rate = starting_power - self.prop_loss >= self.conf['receptionThreshold']
        return reception_rate


class GraphPlotter:
    """
    The graphics plotter class which visualises calculated metrics
    """
    def __init__(self, conf: Dict[str, any], dists: pd.DataFrame, prop_loss: pd.DataFrame,
                 signal_reception: pd.DataFrame):
        self.conf = conf
        self.dists = dists
        self.prop_loss = prop_loss
        self.signal_reception = signal_reception
        self.prop_loss_dict = {}
        self.signal_reception_dict = {}

    def process_row(self, dist_row: pd.Series, timestamp: float) -> None:
        """
        Match plot data with its distance in a single row
        :param dist_row: Row of a dists dataframe
        :param timestamp: The timestamp of a given row
        """
        v0 = dist_row.name
        for v1, dist in dist_row.items():
            if not np.isnan(dist):
                prop_loss = self.prop_loss[v1][timestamp, v0]
                signal_reception = self.signal_reception[v1][timestamp, v0]
                prop_loss_dist = int(np.floor(dist / self.conf['propLossRegion']) * self.conf['propLossRegion'])
                packages_dist = int(np.floor(dist / self.conf['packagesRegion']) * self.conf['packagesRegion'])
                self.prop_loss_dict[prop_loss_dist].append(prop_loss)
                self.signal_reception_dict[packages_dist]['correct'] += signal_reception
                self.signal_reception_dict[packages_dist]['total'] += 1

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

    def generate_distance_data_dependencies(self) -> None:
        """
        Match plot data with its distance
        """
        prop_loss_labels = self.generate_distance_regions(self.conf['propLossRegion'])
        packages_labels = self.generate_distance_regions(self.conf['packagesRegion'])
        self.prop_loss_dict = {label: [] for label in prop_loss_labels}
        self.signal_reception_dict = {label: {'correct': 0, 'total': 0} for label in packages_labels}
        for timestamp, new_df in self.dists.groupby(level=0):
            new_df = new_df.droplevel(0)
            new_df.apply(self.process_row, axis=1, args=(timestamp,))

    def plot_metrics(self) -> None:
        """
        Plot the simulation metrics using matplotlib
        """
        self.generate_distance_data_dependencies()
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        prop_loss_x, prop_loss_y = self.plot_prop_loss()
        packet_rec_x, packet_rec_y = self.plot_prr()
        packet_loss_x, packet_loss_y = self.plot_plr()
        fig.suptitle('Simulation metrics')
        ax0.set_title('PL (propagation loss)')
        ax0.plot(prop_loss_x, prop_loss_y)
        ax1.set_title('PRR (packet reception ratio)')
        ax1.plot(packet_rec_x, packet_rec_y)
        ax2.set_title('PLR (packet loss ratio)')
        ax2.plot(packet_loss_x, packet_loss_y)
        plt.show()

    def plot_prop_loss(self) -> Tuple[List[int], List[float]]:
        """
        Plot propagation loss values averaged by distance
        """
        plot_x, plot_y = [], []
        for key in self.prop_loss_dict:
            if self.prop_loss_dict[key]:
                plot_x.append(key)
                plot_y.append(np.mean(self.prop_loss_dict[key]))
        return plot_x, plot_y

    def plot_prr(self) -> Tuple[List[int], List[float]]:
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

    def plot_plr(self) -> Tuple[List[int], List[float]]:
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


if __name__ == '__main__':
    input_handler = UserInput()
    data = input_handler.start()
    if data:
        calc_proc = CalcProc(data[0], data[1])
        plotter = GraphPlotter(data[0], calc_proc.dists, calc_proc.prop_loss, calc_proc.reception_rate)
        plotter.plot_metrics()

