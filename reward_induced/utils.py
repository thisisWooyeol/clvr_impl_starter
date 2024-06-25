import logging

def _setup_logger(log_file_path):
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(log_file_path, mode='w')
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)
    return logger