import logging

logging.basicConfig(format='[%(asctime)s] logging.BASIC_FORMAT', datefmt='%a, %d %b %Y %H:%M:%S') # Avoid "no handlers could be found" warnings
logger = logging.getLogger('LFD')

# Seems like there should be a simpler way...
# See: https://docs.python.org/2/howto/logging-cookbook.html
ch = logging.StreamHandler()   # Console logging
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s]:%(levelname)s:%(module)s:%(funcName)s:  %(message)s', '%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.propagate = False

logger.setLevel(logging.DEBUG)

#from chroma.log import logger as chroma_logger
#chroma_logger.setLevel(logging.DEBUG)
