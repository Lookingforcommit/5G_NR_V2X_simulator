import pandas as pd
import os
import re
from typing import Dict, Any, Tuple
import exceptions


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
    NECESSARY_PARAMETERS = VALID_PARAMETERS.keys() - DEFAULT_PARAMETERS.keys()
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
    VALID_SCENARIOS = {'highway'}

    def __init__(self):
        pass

    def read_config_file(self, filename: str) -> Dict[str, Any]:
        """
        Check if the config file is not corrupted and parse it into a dict
        :param filename: Path to configuration file
        :return: Parsed config dict
        :raises exceptions.IncorrectConfigFormatException: If config file has incorrect format
        :raises exceptions.IncorrectConfigParameterException: If config file contains invalid parameters
        :raises exceptions.DuplicateConfigParameterException: If config file contains duplicate parameters
        :raises exceptions.MissingConfigParameterException: If some of the required parameters are missing
        """
        with open(filename) as f:
            lines = f.readlines()
        config_params = {}
        reg_check = '[^ =]*=[^ =]*'
        for i in range(len(lines)):
            line = lines[i].strip()
            if not re.fullmatch(reg_check, line):
                raise exceptions.IncorrectConfigFormatException(str(i + 1), line)
            param, val = line.split('=')
            if param not in self.VALID_PARAMETERS:
                raise exceptions.IncorrectConfigParameterException(param, list(self.VALID_PARAMETERS.keys()))
            if param in config_params:
                raise exceptions.DuplicateConfigParameterException(param, str(i + 1))
            config_params[param] = val
        missing_params = self.NECESSARY_PARAMETERS - config_params.keys()
        if missing_params:
            raise exceptions.MissingConfigParameterException(list(missing_params))
        for default_param in self.DEFAULT_PARAMETERS:
            if default_param not in config_params:
                config_params[default_param] = self.DEFAULT_PARAMETERS[default_param]
        return config_params

    def check_config_parameters(self, dct: Dict[str, Any]) -> None:
        """
        Check if provided config parameters are correct. Should be called after parsing the config file.
        :param dct: Config parameters
        :raises exceptions.IncorrectConfigParameterTypeException: If config parameter has incorrect type
        :raises exceptions.IncorrectSimDataPathException: If simulation data path is incorrect
        :raises exceptions.IncorrectSimScenarioException: If simulation scenario is invalid
        """
        for param in dct:
            if self.VALID_PARAMETERS[param] != str:
                try:
                    dct[param] = self.VALID_PARAMETERS[param](dct[param])
                except ValueError:
                    raise exceptions.IncorrectConfigParameterTypeException(param, str(self.VALID_PARAMETERS[param]))
        if not os.path.isfile(dct['dataFilepath']):
            raise exceptions.IncorrectSimDataPathException
        if dct['envScenario'] not in self.VALID_SCENARIOS:
            raise exceptions.IncorrectSimScenarioException(dct['envScenario'], list(self.VALID_SCENARIOS))

    @staticmethod
    def check_duplicate_columns(filename: str, ext: str, configs: Dict[str, Any]) -> None:
        """
        Check if a simulation data file contains duplicate columns
        :param filename: Path to the simulation data file
        :param ext: The extension of the simulation data file
        :param configs: Configuration data
        :raises exceptions.DuplicateSimDataColumnsException: If duplicate columns are found
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
                    raise exceptions.DuplicateSimDataColumnsException()

    def read_sim_datafile(self, filename: str, configs: Dict[str, Any]) -> pd.DataFrame:
        """
        Read a simulation data file and parse it to a pandas dataframe
        :param filename: Path to the simulation data file
        :param configs: Configuration data
        :return: Parsed pandas dataframe
        :raises exceptions.IncorrectSimDataExtException: If simulation data file has unsupported extension
        :raises exceptions.IncorrectSimDataFormatException: If simulation data file has incorrect format
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.PANDAS_READERS:
            raise exceptions.IncorrectSimDataExtException(list(self.PANDAS_READERS.keys()))
        # noinspection PyTypedDict
        reader = self.PANDAS_READERS[ext]
        self.check_duplicate_columns(filename, ext, configs)
        try:
            if ext == '.csv':
                df = reader(filename, sep=configs['csvSep'], dtype=self.VALID_DATA_COLUMNS,
                            usecols=self.VALID_DATA_COLUMNS)
            else:
                df = reader(filename, dtype=self.VALID_DATA_COLUMNS, usecols=self.VALID_DATA_COLUMNS)
        except Exception as exc:
            raise exceptions.IncorrectSimDataFormatException(str(exc))
        return df

    def start(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Start the user dialogue
        :return: Parsed configuration data, simulation data
        :raises exceptions.DataParsingException: If any errors appeared during parsing
        """
        print('Введите путь к файлу конфигурации: ', end='')
        conf = input()
        while not os.path.isfile(conf):
            print('Введён некорректный путь к файлу конфигурации, повторите ввод: ', end='')
            conf = input()
        config_params = self.read_config_file(conf)
        self.check_config_parameters(config_params)
        sim_data = config_params['dataFilepath']
        sim_data = self.read_sim_datafile(sim_data, config_params)
        return config_params, sim_data
