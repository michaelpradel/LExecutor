import logging


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger("LExecutor logger")
logger.setLevel(logging.INFO)

logger.info("Logging starts")