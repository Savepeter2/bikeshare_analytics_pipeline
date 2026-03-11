import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


log_file_name = os.getenv('log_file_name')
error_log_file_name = os.getenv('error_log_file_name')
var_dir = os.getenv('var_dir')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_RAW_BUCKET = os.getenv('S3_RAW_BUCKET')
RAW_S3_KEY = os.getenv('RAW_S3_KEY')
TRANSFORMED_BUCKET = os.getenv('TRANSFORMED_BUCKET')
TRANSFORMED_BUCKET_FOLDER = os.getenv('TRANSFORMED_BUCKET_FOLDER')
TRANSFORMED_BUCKET_METADATA_PREFIX = os.getenv('TRANSFORM_BUCKET_METADATA_PREFIX')
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_BUCKET_FOLDER = os.getenv('RAW_BUCKET_FOLDER')
RAW_BUCKET_METADATA_PREFIX = os.getenv('RAW_BUCKET_METADATA_PREFIX')
RAW_BUCKET_WEEKLY_DUMP_PREFIX = os.getenv('RAW_BUCKET_WEEKLY_DUMP_PREFIX')
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USERNAME = os.getenv('SNOWFLAKE_USERNAME')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')
SNOWFLAKE_ROLE = os.getenv('SNOWFLAKE_ROLE')
STAGE_NAME = os.getenv('STAGE_NAME')
SOURCE_BUCKET = os.getenv('SOURCE_BUCKET')
SOURCE_S3_KEY = os.getenv('SOURCE_S3_KEY')
AIRFLOW_MAIL_USERS = os.getenv('AIRFLOW_MAIL_USERS')
AIRFLOW_MAIL_SUBJECT_TEMPLATE = os.getenv('AIRFLOW_MAIL_SUBJECT_TEMPLATE')
AIRFLOW_MAIL_HTML_TEMPLATE = os.getenv('AIRFLOW_MAIL_HTML_TEMPLATE')
SLACK_BOT_OAUTH_TOKEN = os.getenv('SLACK_BOT_OAUTH_TOKEN')
SLACK_CHANNEL_ID = os.getenv('SLACK_CHANNEL_ID')
SLACK_BOT_NAME = os.getenv('SLACK_BOT_NAME')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))




S3_CONFIG = {
            'raw_bucket': S3_RAW_BUCKET,
            'source_bucket': SOURCE_BUCKET,
            'source_s3_key': SOURCE_S3_KEY,
            'access_key': AWS_ACCESS_KEY,
            'secret_key': AWS_SECRET_KEY,
            'transformed_bucket_metadata_prefix': TRANSFORMED_BUCKET_METADATA_PREFIX,
            'transformed_bucket': TRANSFORMED_BUCKET,
            'transformed_bucket_folder': TRANSFORMED_BUCKET_FOLDER,
            'raw_s3_key': RAW_S3_KEY,
            'raw_bucket_folder': RAW_BUCKET_FOLDER,
            'raw_bucket_metadata_prefix': RAW_BUCKET_METADATA_PREFIX,
            'raw_bucket_weekly_dump_prefix': RAW_BUCKET_WEEKLY_DUMP_PREFIX
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