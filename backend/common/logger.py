import logging
import sys


LOG_PATTERN = '%(asctime)s: %(filename)s:%(lineno)d - %(levelname)s - %(message)s'


def _new_basic_formatter():
    return logging.Formatter(fmt=LOG_PATTERN)


def new_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_new_basic_formatter())

    logger.addHandler(console_handler)

    return logger


class CustomPrefixLogger:
    """
    Simple wrapper used to print a string prefix on the message
    """
    base_log: logging.Logger
    prefix: str

    def __init__(self, base_log: logging.Logger, prefix: str) -> None:
        self.base_log = base_log
        self.prefix = prefix

    def info(self, msg: str):
        self.base_log.info(f'[{self.prefix}] {msg}')

    def error(self, msg: str):
        self.base_log.error(f'[{self.prefix}] {msg}')

    def warning(self, msg: str):
        self.base_log.warning(f'[{self.prefix}] {msg}')
