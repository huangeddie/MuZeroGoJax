"""Logging utilities."""
import datetime

_START_TIME = None


def initialize_start_time():
    """Initialize the start time for the logger."""
    global _START_TIME  # pylint: disable=global-statement
    _START_TIME = datetime.datetime.now().replace(microsecond=0)


def log(msg: str):
    """Prints a message with the time elapsed since the start of the program."""
    print(f'{datetime.datetime.now().replace(microsecond=0) - _START_TIME} | '
          f'{msg}')
