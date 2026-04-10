import logging


# 获取日志记录器
logger = logging.getLogger("demo04_logging")

# 创建控制台和文件处理器
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("demo04_logging.log",mode="a",encoding="utf-8")

# 设置日志级别
logger.setLevel(logging.DEBUG)
# 设置formatter格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 添加文件和控制台处理器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 不同日志级别
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")