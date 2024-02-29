import pandas as pd
import numpy as np
import os
import re
from typing import Dict, Any, Tuple, Optional
from copy import copy
from scipy.spatial.distance import pdist, squareform


class UserInput:
    VALID_PARAMETERS = {
        'dataFilepath': str,
        'csvSep': str,
        'signalFreq': int,
        'envScenario': str,
    }
    DEFAULT_PARAMETERS = {
        'csvSep': ';',
        'envScenario': 'highway',
    }
    NECESSARY_PARAMETERS = {'dataFilepath', 'signalFreq'}
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
        ans = True
        for param in dct:
            if param not in self.DEFAULT_PARAMETERS:
                correct = True
                if self.VALID_PARAMETERS[param] == int:
                    correct = dct[param].isnumeric()
                    if correct:
                        dct[param] = int(dct[param])
                if not correct:
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
        with open(filename) as f:
            lines = f.readlines()
        config_params = copy(self.DEFAULT_PARAMETERS)
        reg_check = '[^ =]*=[^ =]*'
        correct_params = True
        for i in range(len(lines)):
            line = lines[i].strip()
            if not re.fullmatch(reg_check, line):
                print(f'Некорректный формат файла конфигурации, ошибка в строке {i + 1}', line, sep='\n', end='')
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
            print(f'В конфигурационном файле отсуствуют обязательные параметры симуляции: {missing_params}')
            correct_params = False
        else:
            correct_params = correct_params and self.check_config_parameters(config_params)
        return correct_params, config_params

    @staticmethod
    def check_duplicate_columns(filename: str, ext: str, configs: Dict[str, Any]) -> bool:
        if ext == '.csv':
            first_row = pd.read_csv(filename, nrows=1, sep=configs['csvSep'])
        else:
            first_row = pd.read_csv(filename, nrows=1)
        columns = list(first_row.columns)
        for i in range(len(columns)):
            col_name1 = columns[i]
            for j in range(i + 1, len(columns)):
                col_name2 = columns[j]
                if col_name2.startswith(col_name1):
                    return True
        return False

    def read_sim_datafile(self, filename: str, configs: Dict[str, Any]) -> Tuple[bool, pd.DataFrame]:
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
    LOGNORM_MEAN = 0
    LOGNORM_SIGMA = 3
    CAR_WIDTH = 1.5  # meters
    CAR_LENGTH = 4  # meters
    FLOAT_INACCURACY = 0.01
    Point = Tuple[float, float]

    def __init__(self, conf: Dict[str, Any], sim_data: pd.DataFrame):
        self.cords = sim_data
        self.conf = conf
        self.preprocess_data()
        self.dists = self.calc_dists()
        self.prop_loss = self.calc_prop_loss()

    def preprocess_data(self) -> None:
        self.cords.rename(columns={'timestep_time': 'timestamp'}, inplace=True)
        self.cords = self.cords.dropna().sort_values(['timestamp', 'vehicle_id'])
        self.cords['vehicle_id'] = self.cords['vehicle_id'].str.replace('.', '_')
        self.cords.set_index(['timestamp', 'vehicle_id'], inplace=True)
        new_cords = self.cords.apply(lambda x: self.get_full_veh_cords(**x), axis=1, result_type='expand')
        self.cords = pd.concat([self.cords, new_cords], axis=1)

    def parallel_segments(self, s0_p0: Point, s0_p1: Point, s1_p0: Point, s1_p1: Point) -> bool:
        if s0_p1[1] == s0_p0[1] or s1_p1[1] == s1_p0[1]:
            parallel = s0_p1[1] == s0_p0[1] and s1_p1[1] == s1_p0[1]
        else:
            k0 = (s0_p1[0] - s0_p0[0]) / (s0_p1[1] - s0_p0[1])
            k1 = (s1_p1[0] - s1_p0[0]) / (s1_p1[1] - s1_p0[1])
            parallel = abs(k1 - k0) <= self.FLOAT_INACCURACY
        return parallel

    def segment_intersection(self, s0_p0: Point, s0_p1: Point, s1_p0: Point, s1_p1: Point) -> bool:
        parallel = self.parallel_segments(s0_p0, s0_p1, s1_p0, s1_p1)
        if parallel:
            p0_dist = np.sqrt((s0_p0[0] - s1_p0[0]) ** 2 + (s0_p0[1] - s1_p0[1]) ** 2)
            p1_dist = np.sqrt((s0_p1[0] - s1_p0[0]) ** 2 + (s0_p1[1] - s1_p0[1]) ** 2)
            s0_len = np.sqrt((s0_p1[0] - s0_p0[0]) ** 2 + (s0_p1[1] - s0_p0[1]) ** 2)
            return abs(p0_dist + p1_dist - s0_len) <= self.FLOAT_INACCURACY
        else:
            matrix = np.array([
                [s0_p1[0] - s0_p0[0], s1_p0[0] - s1_p1[0]],
                [s0_p1[1] - s0_p0[1], s1_p0[1] - s1_p1[1]]
            ])
            solution = np.array([
                s1_p0[0] - s0_p0[0],
                s1_p0[1] - s0_p0[1]
            ])
            a, b = np.linalg.solve(matrix, solution)
            ans = 0 <= a <= 1 and 0 <= b <= 1
            return ans

    def get_full_veh_cords(self, vehicle_x: float, vehicle_y: float, vehicle_angle: float, **kwargs) -> dict:
        vehicle_angle = 90 - vehicle_angle  # convert to trigonometric
        if vehicle_angle > 180:
            vehicle_angle -= 180
        angle_sin, angle_cos = np.sin(vehicle_angle), np.cos(vehicle_angle)
        top_l_x = (vehicle_x + self.CAR_LENGTH / 2 * angle_cos - self.CAR_WIDTH / 2 * angle_sin)
        top_l_y = (vehicle_y + self.CAR_LENGTH / 2 * angle_sin + self.CAR_WIDTH / 2 * angle_cos)
        top_r_x = (vehicle_x + self.CAR_LENGTH / 2 * angle_cos + self.CAR_WIDTH / 2 * angle_sin)
        top_r_y = (vehicle_y + self.CAR_LENGTH / 2 * angle_sin - self.CAR_WIDTH / 2 * angle_cos)
        bot_l_x = (vehicle_x - self.CAR_LENGTH / 2 * angle_cos - self.CAR_WIDTH / 2 * angle_sin)
        bot_l_y = (vehicle_y - self.CAR_LENGTH / 2 * angle_sin + self.CAR_WIDTH / 2 * angle_cos)
        bot_r_x = (vehicle_x - self.CAR_LENGTH / 2 * angle_cos + self.CAR_WIDTH / 2 * angle_sin)
        bot_r_y = (vehicle_y - self.CAR_LENGTH / 2 * angle_sin - self.CAR_WIDTH / 2 * angle_cos)
        ans = {
            'top_l_x': top_l_x,
            'top_l_y': top_l_y,
            'top_r_x': top_r_x,
            'top_r_y': top_r_y,
            'bot_l_x': bot_l_x,
            'bot_l_y': bot_l_y,
            'bot_r_x': bot_r_x,
            'bot_r_y': bot_r_y
        }
        return ans

    def calc_highway_los_nlosv(self, dists: pd.DataFrame) -> pd.DataFrame:
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.log10(np.full(dists.shape, self.conf['signalFreq']))
        prop_loss = 32.4 + 20 * (np.log10(dists) + np.log10(signal_freq)) + lognorm
        return prop_loss

    def calc_urban_los_nlosv(self, dists: pd.DataFrame) -> pd.DataFrame:
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.full(dists.shape, self.conf['signalFreq'])
        prop_loss = 38.77 + 16.7 * np.log10(dists) + 18.2 * np.log10(signal_freq) + lognorm
        return prop_loss

    def calc_urban_nlos(self, dists: pd.DataFrame) -> pd.DataFrame:
        lognorm = np.random.lognormal(self.LOGNORM_MEAN, self.LOGNORM_SIGMA, size=dists.shape)
        signal_freq = np.full(dists.shape, self.conf['signalFreq'])
        prop_loss = 36.85 + 30 * np.log10(dists) + 18.9 * np.log10(signal_freq) + lognorm
        return prop_loss

    def calc_prop_loss(self) -> Optional[pd.DataFrame]:
        ans_df = None
        if self.conf['envScenario'] == 'highway':
            ans_df = self.calc_highway_los_nlosv(self.dists)
        return ans_df

    def find_los_blockage(self) -> pd.DataFrame:
        ans_df = pd.DataFrame(False, index=self.dists.index, columns=self.dists.columns)
        for timestamp, new_df in self.cords.groupby(level=0):
            new_df = new_df.droplevel(0)
            rows = new_df.itertuples()
            for v0 in rows:
                v0_pos = (v0.vehicle_x, v0.vehicle_y)
                for v1 in rows:
                    v1_pos = (v1.vehicle_x, v1.vehicle_y)
                    if v1[0] != v0[0]:
                        for block_v in rows:
                            if block_v[0] != v0[0] and block_v[0] != v1[0]:
                                bot_left = (block_v.bot_l_x, block_v.bot_l_y)
                                top_left = (block_v.top_l_x, block_v.top_l_y)
                                top_right = (block_v.top_r_x, block_v.top_r_y)
                                bot_right = (block_v.bot_r_x, block_v.bot_r_y)
                                block_left = self.segment_intersection(v0_pos, v1_pos, bot_left, top_left)
                                block_top = self.segment_intersection(v0_pos, v1_pos, top_left, top_right)
                                block_right = self.segment_intersection(v0_pos, v1_pos, bot_right, top_right)
                                block_bot = self.segment_intersection(v0_pos, v1_pos, bot_left, bot_right)
                                if any([block_left, block_top, block_right, block_bot]):
                                    ans_df[v1[0]][timestamp, v0[0]] = True
                                    break
        return ans_df

    def calc_dists(self) -> pd.DataFrame:
        def transform(df: pd.DataFrame) -> Optional[pd.DataFrame]:
            if df.shape[0] > 1:
                dists = squareform(pdist(list(zip(df['vehicle_x'], df['vehicle_y']))))
                new_df = pd.DataFrame(dists, index=df.index.get_level_values(1), columns=df.index.get_level_values(1))
                return new_df

        dists = self.cords.groupby(level=0).apply(transform)
        dists.mask(np.isclose(dists, 0), inplace=True)
        return dists


input_handler = UserInput()
data = input_handler.start()
if data:
    calc_proc = CalcProc(data[0], data[1])
    prop_loss = calc_proc.prop_loss
    print(prop_loss)
