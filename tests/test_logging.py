from advpipe.log import logger


def test_always_pass():
    logger.debug("This is debug msg")
    logger.info("This is info msg")
    logger.warning("This is warning msg")