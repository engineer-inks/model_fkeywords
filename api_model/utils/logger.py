from api_model.utils.colorize import Colorize
from datetime import datetime
from pytz import timezone


sao_paulo_timezone = timezone('America/Sao_Paulo')


class Logger:
    """
    Logger class

    print only logs with criticity_level equal or less level informed
    """

    current_level = 5
    levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']

    def __init__(self, level='NOTSET'):
        self.set_level(level)

    def set_level(self, level):
        try:
            self.current_level = self.levels.index(level)
        except ValueError:
            self.error(f'{level} is not a accepted level. Try one of this: {self.levels}')

    @staticmethod
    def _execute(msg):
        print(f'[{datetime.now().astimezone(sao_paulo_timezone).strftime("%m/%d %H:%M:%S")}] - {msg}')

    def debug(self, msg):
        if self.levels.index('DEBUG') <= self.current_level:
            self._execute(f"{Colorize.get_color('[Debug]: ', color='green', style='bold')} {msg}")

    def info(self, msg):
        if self.levels.index('INFO') <= self.current_level:
            self._execute(f"{Colorize.get_color('[Info]: ', color='cyan', style='bold')} {msg}")

    def warn(self, msg):
        if self.levels.index('WARNING') <= self.current_level:
            self._execute(f"{Colorize.get_color('[Warning]: ', color='yellow', style='bold')} {msg}")

    def error(self, msg):
        if self.levels.index('ERROR') <= self.current_level:
            self._execute(f"{Colorize.get_color('[Error]: ', color='red', style='blink')} {msg}")

    def critical(self, msg):
        if self.levels.index('CRITICAL') <= self.current_level:
            self._execute(f"{Colorize.get_color('[CRITICAL]: ', color='on_red', style='blink')} {msg}")


logger = Logger(level='DEBUG')

__all__ = ['logger']
