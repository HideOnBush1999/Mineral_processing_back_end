import logging.config
import codecs

def configure_logging():
    with codecs.open('./logging.conf', 'r', 'utf-8') as f:
        logging.config.fileConfig(f)
    return logging.getLogger()

logger = configure_logging()
