# import os
# import sys
# from typing import Tuple
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from configs.config import log_file_path, error_log_file_path

# import logging

# logging.basicConfig(
#     filename=log_file_path,
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s %(message)s",
# )


# error_logger = logging.getLogger("error_logger")
# error_logger.setLevel(logging.ERROR)
# error_handler = logging.FileHandler(error_log_file_path)
# error_logger.addHandler(error_handler)

# logger_handler = logging.FileHandler(log_file_path)
# logger = logging.getLogger("logger")
# logger.addHandler(logger_handler)
# logger.setLevel(logging.INFO)

# # Console handler for main logger (THIS IS WHAT YOU'RE MISSING)
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)


import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import log_file_path, error_log_file_path
import logging

# Main logger - writes to file AND console
logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

# File handler for main logger
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Console handler for main logger (THIS IS WHAT YOU'RE MISSING)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Error logger - writes to error file AND console
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)

# File handler for error logger
error_file_handler = logging.FileHandler(error_log_file_path)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

# Console handler for error logger
error_console_handler = logging.StreamHandler(sys.stderr)
error_console_handler.setLevel(logging.ERROR)
error_console_handler.setFormatter(formatter)

# Add both handlers to error logger
error_logger.addHandler(error_file_handler)
error_logger.addHandler(error_console_handler)
