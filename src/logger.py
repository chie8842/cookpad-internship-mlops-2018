import logging
import datetime


def logger_config(log_name):
    log_file = "log.txt"

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)s %(message)s')
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))

    logging.getLogger().addHandler(stream_handler)
    logging.getLogger().addHandler(file_handler)

    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(log_name)

    return logger
