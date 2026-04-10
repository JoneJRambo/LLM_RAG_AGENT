import logging


# 基本配置
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="demo04_logging.log")

# 日志记录器
logger = logging.getLogger("demo04_logging")

# 不同日志级别
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

