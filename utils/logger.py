import logging


class Logger:
    logger = None

    def __init__(self, log_name, log_file_path, log_format):
        if Logger.logger is not None:
            return

        Logger.logger = logging.getLogger(log_name)
        Logger.logger.setLevel(level=logging.DEBUG)
        file_handler = logging.FileHandler(filename=log_file_path)
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(fmt=logging.Formatter(log_format))
        Logger.logger.addHandler(file_handler)


def debug(message):
    Logger.logger.debug(message)


def info(message):
    Logger.logger.info(message)


def warning(message):
    Logger.logger.warning(message)


def error(message):
    Logger.logger.error(message)


def critical(message):
    Logger.logger.critical(message)
