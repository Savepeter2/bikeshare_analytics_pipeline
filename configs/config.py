import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configparser
from airflow.models import Variable

config = configparser.RawConfigParser()
secrets_config = config.read('secrets.ini')

LAST_ROW_INDEX = Variable.get("last_row_index", default_var=0)
LAST_ROW_INDEX = int(LAST_ROW_INDEX)
BATCH_SIZE = Variable.get("batch_size", default_var=20)
# LAST_ROW_INDEX = 0
# BATCH_SIZE = 5
log_file_name = config['LOGGING']['log_file_name']
error_log_file_name = config['LOGGING']['error_log_file_name']
var_dir = config['LOGGING']['var_dir']
AWS_ACCESS_KEY = config['AWS_CREDENTIALS']['AWS_ACCESS_KEY']
AWS_SECRET_KEY = config['AWS_CREDENTIALS']['AWS_SECRET_ACCESS_KEY']
AWS_REGION = config['AWS_CREDENTIALS']['AWS_REGION']
S3_RAW_BUCKET = config['AWS_CREDENTIALS']['S3_RAW_BUCKET']
RAW_S3_KEY = config['AWS_CREDENTIALS']['RAW_S3_KEY']
TRANSFORMED_BUCKET = config['AWS_CREDENTIALS']['TRANSFORMED_BUCKET']
RAW_DATA_PATH = config['LOCAL']['RAW_DATA_PATH']
SNOWFLAKE_ACCOUNT = config['SNOWFLAKE']['snowflake_account']
SNOWFLAKE_USERNAME = config['SNOWFLAKE']['snowflake_username']
SNOWFLAKE_PASSWORD = config['SNOWFLAKE']['snowflake_password']
SNOWFLAKE_WAREHOUSE = config['SNOWFLAKE']['snowflake_warehouse']
SNOWFLAKE_DATABASE = config['SNOWFLAKE']['snowflake_database']
SNOWFLAKE_SCHEMA = config['SNOWFLAKE']['snowflake_schema']
SNOWFLAKE_ROLE = config['SNOWFLAKE']['snowflake_role']
STAGE_NAME = config['SNOWFLAKE']['stage_name']
SOURCE_BUCKET = config['AWS_CREDENTIALS']['SOURCE_BUCKET']
SOURCE_S3_KEY = config['AWS_CREDENTIALS']['SOURCE_S3_KEY']
AIRFLOW_MAIL_USERS = config['AIRFLOW']['MAIL_USERNAMES']
AIRFLOW_MAIL_SUBJECT_TEMPLATE = config['AIRFLOW_EMAIL']['AIRFLOW__EMAIL__SUBJECT_TEMPLATE']
AIRFLOW_MAIL_HTML_TEMPLATE = config['AIRFLOW_EMAIL']['AIRFLOW__EMAIL__HTML_CONTENT_TEMPLATE']

S3_CONFIG = {
            'transformed_bucket': TRANSFORMED_BUCKET,
            'raw_bucket': S3_RAW_BUCKET,
            'access_key': AWS_ACCESS_KEY,
            'secret_key': AWS_SECRET_KEY,
            'base_prefix': TRANSFORMED_BUCKET,
            'raw_s3_key': RAW_S3_KEY
        }
SNOWFLAKE_CONFIG = {
    'snowflake_account': SNOWFLAKE_ACCOUNT,
    'snowflake_username': SNOWFLAKE_USERNAME,
    'snowflake_password': SNOWFLAKE_PASSWORD,
    'snowflake_warehouse': SNOWFLAKE_WAREHOUSE,
    'snowflake_database': SNOWFLAKE_DATABASE,
    'snowflake_schema': SNOWFLAKE_SCHEMA,
    'snowflake_role': SNOWFLAKE_ROLE,
    'stage_name': STAGE_NAME
}

class LogFileFormatError(ValueError):
    """Raised when the log file format is invalid"""

    pass


class InvalidInputTypeError(ValueError):
    """Raised when input arguments are not strings"""

    pass


def create_log_file(
    log_file_name: str, error_log_file_name: str, var_dir: str
) -> Tuple[str, str]:
    """
    Function to create a log file

    Args:
    log_file_name (str): Name of the log file
    error_log_file_name (str): Name of the error log file
    var_dir (str): Name of the directory to store the log files

    Returns:
    Tuple[str, str]: A tuple containing the path to the log file
    and the path to the error log file

    """

    if (
        isinstance(log_file_name, str)
        and isinstance(error_log_file_name, str)
        and isinstance(var_dir, str)
    ):
        if log_file_name.endswith(".log") and error_log_file_name.endswith(".log"):
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(curr_dir)
            var_dir = os.path.join(base_dir, var_dir)

            if not os.path.exists(var_dir):
                os.makedirs(var_dir)
            log_dir = os.path.join(var_dir, "log")

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_file_path = os.path.join(log_dir, log_file_name)
            if not os.path.exists(log_file_path):
                with open(log_file_path, "w") as f:
                    f.write("")

            error_log_file_path = os.path.join(log_dir, error_log_file_name)

            if not os.path.exists(error_log_file_path):
                with open(error_log_file_path, "w") as f:
                    f.write("")

            return log_file_path, error_log_file_path
        else:
            raise LogFileFormatError("Invalid file format. Only log files are allowed")

    else:
        raise InvalidInputTypeError(
            "Invalid input arguments. Input arguments must be strings"
        )

log_file_path, error_log_file_path = create_log_file(
    log_file_name, error_log_file_name, var_dir
)

