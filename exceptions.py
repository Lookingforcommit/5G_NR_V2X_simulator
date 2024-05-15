from typing import List
from abc import abstractmethod


class DataParsingException(Exception):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class ConfigException(DataParsingException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class SimDataException(DataParsingException):
    def __init__(self):
        pass

    def __str__(self):
        pass


class IncorrectConfigFormatException(ConfigException):
    def __init__(self, incorrect_line_number: str, incorrect_line: str):
        self.error_message = (f'Некорректный формат файла конфигурации, ошибка в строке {incorrect_line_number}\n'
                              f'{incorrect_line}')

    def __str__(self):
        return self.error_message


class IncorrectConfigParameterException(ConfigException):
    def __init__(self, incorrect_param: str, valid_params: List[str]):
        error_message = f'Некорректный параметр симуляции {incorrect_param}, доступные опции:'
        valid_params = '\n'.join(valid_params)
        self.error_message = '\n'.join([error_message, valid_params])

    def __str__(self):
        return self.error_message


class DuplicateConfigParameterException(ConfigException):
    def __init__(self, dup_param: str, incorrect_line_number: str):
        self.error_message = f'Повторяющийся параметр симуляции {dup_param} в строке {incorrect_line_number}'

    def __str__(self):
        return self.error_message


class MissingConfigParameterException(ConfigException):
    def __init__(self, missing_params: List[str]):
        error_message = f'В конфигурационном файле отсутствуют обязательные параметры симуляции:'
        missing_params = '\n'.join(missing_params)
        self.error_message = '\n'.join([error_message, missing_params])

    def __str__(self):
        return self.error_message


class IncorrectConfigParameterTypeException(ConfigException):
    def __init__(self, incorrect_param: str, correct_type: str):
        self.error_message = f'Некорректный тип параметра {incorrect_param}, требуется {correct_type}'

    def __str__(self):
        return self.error_message


class IncorrectSimDataPathException(SimDataException):
    def __init__(self):
        self.error_message = 'Введён некорректный путь к файлу данных симуляции'

    def __str__(self):
        return self.error_message


class IncorrectSimScenarioException(SimDataException):
    def __init__(self, incorrect_scenario: str, valid_options: List[str]):
        error_message = f'Введён некорректный сценарий симуляции {incorrect_scenario}, доступные опции:'
        valid_options = '\n'.join(valid_options)
        self.error_message = '\n'.join([error_message, valid_options])

    def __str__(self):
        return self.error_message


class IncorrectSimDataFormatException(SimDataException):
    def __init__(self, parser_message: str):
        error_message = 'Некорректный формат файла данных симуляции, сообщение парсера pandas:'
        self.error_message = '\n'.join([error_message, parser_message])

    def __str__(self):
        return self.error_message


class IncorrectSimDataExtException(SimDataException):
    def __init__(self, valid_extensions: List[str]):
        error_message = 'Неподдерживаемое расширение файла данных симуляции, cписок поддерживаемых расширений:'
        valid_extensions = '\n'.join(valid_extensions)
        self.error_message = '\n'.join([error_message, valid_extensions])

    def __str__(self):
        return self.error_message


class DuplicateSimDataColumnsException(SimDataException):
    def __init__(self):
        self.error_message = 'Обнаружены дубликаты в названиях столбцов в файле данных симуляции'

    def __str__(self):
        return self.error_message
