import hashlib
import os
import random
import string
import sys
import re
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Dict, List, Union, Optional, Generator

from configs.logger_config import error_logger, logger
from configs.config import SLACK_BOT_OAUTH_TOKEN, SLACK_CHANNEL_ID, SLACK_BOT_NAME, SNOWFLAKE_CONFIG, S3_CONFIG
from src.models import create_session, AlertsLog
from sqlalchemy import text
import requests
import time
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import boto3
import ast
from typing import Set, Dict
from io import BytesIO
import io
from io import StringIO
from datetime import datetime
import great_expectations as ge
from requests.sessions import Session
from src.schema import raw_data_schema
from configs.config import RAW_DATA_PATH
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class InvalidArgumentTypeError(ValueError):
    """
    Custom exception for handling invalid argument types
    """
    pass

class MissingColumnError(Exception):
    """
    Custom exception for handling missing columns in the dataframe
    """
    pass


def gen_hash_key_station_id(latitude: str,
                             longitude: str) -> Dict[str, str]:
    """
    Function to generate a hash key for a given station, either start or end station
    The hash key is generated using the concatenation of the latitude and longitude of the station
    
    Args:
    latitude(str): The latitude of the station
    longitude(str): The longitude of the station

    Returns:
    Dict[str,str]: A dictionary containing the status of the operation,
                     a message and the hash key

    Example: {
            "status": "success",
            "message": "Hash key generated successfully",
            "hash_key": "c3d8b3d9c4e"
    }
    """

    try:
        if isinstance(latitude, str) and isinstance(longitude, str):
            composite_key = f"{latitude},{longitude}"
            composite_key = composite_key.lower().replace(" ", "")
            hash_object = hashlib.sha256(composite_key.encode())
            hash_key = hash_object.hexdigest()

            logger.info(
                {
                    "status": "success",
                    "message": "Hash key generated successfully",
                    "hash_key": hash_key,
                }
            )

            return {
                "status": "success",
                "message": "Hash key generated successfully for station ID",
                "hash_key": hash_key,
            }

        else:
            raise InvalidArgumentTypeError("Latitude and Longitude for generating station ID must be strings")

    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        except SyntaxError as se:
            exc_error = str(e)
     
        error_logger.error(
            {
                "status": "error",
                "message": "Unable to generate hash key for station ID",
                "error": exc_error,
            }
        )
        
        raise Exception(
            {
                "status": "error",
                "message": "Unable to generate hash key for station ID",
                "error": exc_error,
            }
        )

def extract_source_data(
    aws_access_key: str,
    aws_secret_key: str,
    source_bucket: str,
    source_s3_key: str,
    batch_year: str,
    batch_week: str,
    chunk_size: int,
    raw_data_schema: Dict[str, str]
) -> Dict:
    """
    Extracts data from the source S3 bucket in batches. 

    Args:
        - aws_access_key (str): AWS access key for S3 authentication.
        - aws_secret_key (str): AWS secret key for S3 authentication.
        - source_bucket (str): Name of the source S3 bucket.
        - source_s3_key (str): Key of the source S3 object.
        - batch_year (str): Year of the trips to extract.
        - batch_week (str): Week of the trips to extract.
        - chunk_size (int): Number of rows to extract in each batch.

    Returns:
        - Dict: A dictionary containing the status, message, and the extracted DataFrame.
    """
    try:
        if not all([isinstance(aws_access_key, str),
                    isinstance(aws_secret_key, str),
                    isinstance(source_bucket, str),
                    isinstance(source_s3_key, str)]):
            raise InvalidArgumentTypeError("AWS access key, secret key, source bucket, and source s3 key must be strings")
        if not isinstance(batch_year, str):
            raise InvalidArgumentTypeError("Batch year must be a string")
        if not isinstance(batch_week, str):
            raise InvalidArgumentTypeError("Batch week must be a string")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise InvalidArgumentTypeError("Chunk size must be a positive integer")
        
        s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        

        response = s3_client.get_object(Bucket=source_bucket, Key=source_s3_key)
        source_data = response['Body']
        if not source_data:
            raise Exception({
                "status": "error",
                "message": "No data found in the source s3 path",
                "source_data": source_data
            })

        raw_df = pd.DataFrame()
        for chunk in pd.read_csv(source_data,
                                        dtype=raw_data_schema,
                                        chunksize=chunk_size,
                                        dtype_backend='pyarrow'):
            
            chunk_copy = chunk.copy()
            chunk_copy['year'] = pd.to_datetime(chunk_copy['started_at'], errors='coerce').dt.year.astype(str)
            chunk_copy['week'] = pd.to_datetime(chunk_copy['started_at'], errors='coerce').dt.isocalendar().week.astype(str)
            chunk_copy['month'] = pd.to_datetime(chunk_copy['started_at'], errors='coerce').dt.month.astype(str).str.zfill(2)
            filtered_chunk = chunk[(chunk_copy['year'] == batch_year) & (chunk_copy['week'] == batch_week) \
                                    & (chunk_copy['month'] == '12')]
            raw_df = pd.concat([raw_df, filtered_chunk], ignore_index=True)
            break #only process first chunk for testing purposes

        if raw_df.empty:
            logger.info({
                "status": "success",
                "message": f"No data found for {batch_year}-W{batch_week} in source s3 bucket: {source_bucket}",
                "data": raw_df.shape
            })
            return {
                "status": "success",
                "message": f"No data found for {batch_year}-W{batch_week} in source s3 bucket: {source_bucket}",
                "data": raw_df
            }
        
        logger.info({
            "status": "success",
            "message": f"Successfully extracted {batch_year}-W{batch_week} from source s3 bucket: {source_bucket}",
            "data": raw_df.shape
        })
        return {
            "status": "success",
            "message": f"Successfully extracted {batch_year}-W{batch_week} from source s3 bucket: {source_bucket}",            
            "data": raw_df
        }
        
    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while extracting data from source s3",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while extracting data from source s3",
            "error": str(e)
        })


def validate_raw_data(raw_df: DataFrame) -> Dict:
    """
    This function validates the raw bikeshare data to ensure it meets the required schema and data quality standards.
    
    Args:
        raw_df (DataFrame): The input dataframe containing raw bikeshare trip data.
    
    Returns:
        Dict: A dictionary containing the status and message of the validation operation.
    """

    try:
        if not isinstance(raw_df, DataFrame):
            raise InvalidArgumentTypeError("Input raw_df must be a pandas DataFrame")
        
        ge_df_raw = ge.from_pandas(raw_df)
        required_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id"
        ]

        validate_column_existence = ge_df_raw.expect_table_columns_to_match_set(required_cols)
        print("validate_column_existence", validate_column_existence)
        if not validate_column_existence['success']:
            missing_cols = set(required_cols) - set(ge_df_raw.get_table_columns())
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": f"columns: {sorted(missing_cols)} are missing"
            })
        
        if ge_df_raw.get_row_count() != 0:
        
            validate_ride_id_not_null = ge_df_raw.expect_column_values_to_not_be_null("ride_id")

            if validate_ride_id_not_null['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": "column validation error occured",
                    "error": "ride_id column contains null values"
                })
            
            validate_ride_id_unique = ge_df_raw.expect_column_values_to_be_unique("ride_id")
            if validate_ride_id_unique['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": "column validation error occured",
                    "error": "ride_id column contains duplicate values"
                })
            
            date_time_cols = ["started_at", "ended_at"]
            failing_datetime_cols = []
            for col in date_time_cols:
                validate_datetime_format_result = ge_df_raw.expect_column_values_to_match_strftime_format(col, "%Y-%m-%d %H:%M:%S")
                if validate_datetime_format_result['success'] is False:
                    failing_datetime_cols.append(col)

            if failing_datetime_cols:
                raise Exception({
                    "status": "error",
                    "message": "column type validation error(s) occurred",
                    "error": f"columns with invalid datetime format: {', '.join(failing_datetime_cols)}"
                })

            str_cols = ["ride_id", "rideable_type", "started_at", "ended_at", "member_casual"]
            failing_str_cols = []

            for col in str_cols:
                result = ge_df_raw.expect_column_values_to_be_of_type(col, "str", result_format='SUMMARY')
                
                if not result['success']:
                    failing_str_cols.append(col)

            if failing_str_cols:
                raise Exception({
                    "status": "error",
                    "message": "column type validation error(s) occurred",
                    "error": f"columns with non-string values: {', '.join(failing_str_cols)}"
                })

            float_cols = [
            "start_lat", "start_lng", "end_lat", "end_lng"
            ]
            failing_float_cols = []
            for col in float_cols:
                validate_float_type_result = ge_df_raw.expect_column_values_to_be_of_type(col, "float")

                if not validate_float_type_result['success']:
                    failing_float_cols.append(col)

            if failing_float_cols:
                raise Exception({
                    "status": "error",
                    "message": "column type validation error(s) occurred",
                    "error": f"columns with non-float values: {', '.join(failing_float_cols)}"
                })
    
        logger.info({
            "status": "success",
            "message": "Raw data validation completed successfully",
            "data_shape": raw_df.shape
        })
        return {
            "status": "success",
            "message": "Raw data validation completed successfully",
            "data_shape": raw_df.shape
        }
    
    except Exception as e:
        try:
            gexp_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            gexp_error = str(e)

        error_logger.error({
            "status": "error",
            "message": "An error occurred during raw data validation",
            "error": gexp_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred during raw data validation",
            "error": gexp_error
        })

def requests_session_with_retries() -> Session:
    """
    This function uses the requests library to create a session with retry logic.
    It retries the request up to 3 times with exponential backoff for specific status codes.

    Args:
        None
    
    Returns:
        Session: A requests session with retry logic
    """
    try:
        session = requests.Session()
        retries = Retry(
            total=3,                  
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session
    
    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while creating requests session with retries",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while creating requests session with retries",
            "error": str(e)
        })

def make_request(lat: str, lon: str) -> dict:
    """
    Makes a request to the OpenStreetMap API to fetch the address for the given latitude and longitude.

    Args:
    lat(str): Latitude of the location
    lon(str): Longitude of the location

    Returns:
    dict: A dictionary containing the status, message, and the fetched address
    """

    session = requests_session_with_retries()
    headers = {
                "User-Agent": "BikeshareLocationService/1.0 (contact: saveadekolu@gmail.com)",
                "Accept-Language": "en"
        }
    osm_api_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
    response = session.get(osm_api_url, headers=headers)
    status_code = response.status_code

    if status_code == 200:
        data = response.json()
        address = data.get('display_name', 'Address not found')
        logger.info({
            "status": "success",
            "message": f"Address fetched from OpenStreetMap API successfully for lat: {lat}, lon: {lon}"
        })
        return {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": address
        }

    elif status_code == 429:
        error_logger.error({
            "status": "error",
            "message": "Rate limit exceeded when fetching data from OpenStreetMap API",
            "error": response.text
        })
        raise Exception({
            "status": "error",
            "message": "Rate limit exceeded when fetching data from OpenStreetMap API",
            "error": response.text
        })
    
    else:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while making request to OpenStreetMap API",
            "error": response.text
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while making request to OpenStreetMap API",
            "error": response.text
        })


def get_address(lat: str, lon: str) -> Dict:
    """
    This function fetches the address from OpenStreetMap API using the provided latitude and longitude. 

    Args:
    lat(str): Latitude of the location
    long(str): Longitude of the location

    Returns:
    

    """
    
    try:
        if not isinstance(lat, str) or not isinstance(lon, str):
            raise InvalidArgumentTypeError("Latitude and Longitude must be string types")
           
        return make_request(lat, lon)

    except ConnectionError as ce:
        try:
            ce_error = ast.literal_eval(str(ce))
        
        except SyntaxError as se:
            ce_error = str(ce)
        error_logger.error({"status": "error",
                            "message": "ConnectionError occurred while fetching address from OpenStreetMap API, retrying in 30 seconds...",
                            "error": ce_error
                            })
        time.sleep(15)
        retry_api_response = make_request(lat, lon)
        return retry_api_response
    
    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while fetching address from OpenStreetMap API",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while fetching address from OpenStreetMap API",
            "error": exc_error
        })


def add_station_id(lat: str, lon: str) -> Dict:
    """
    This function adds an ID for both start station ID and end station ID, marking their identifiers

    Args:

    lat(str): Latitude of the station 
    lon(str): Longitude of the station 

    Returns:
    Dict: A dictionary containing the status, message, and the generated station ID

    """
    try:
        if not isinstance(lat, str) or not isinstance(lon, str):
            raise InvalidArgumentTypeError("Latitude and Longitude must be string types")

        response = gen_hash_key_station_id(lat, lon)
        station_id = response['hash_key']

        return {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": station_id
        }

    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while generating station ID",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while generating station ID",
            "error": exc_error
        })


def clean_raw_data(raw_df: DataFrame) -> DataFrame:
    """
    This function cleans the raw bikeshare data by removing duplicates, standardizing formats, and handling missing values.

    It performs the following operations:
    1. Removes duplicate entries based on 'ride_id'.
    2. Standardizes string columns to lowercase and strips leading/trailing whitespace.

    The missing values handling in other columns is done in the impute_missing_station_ids function.
    The datetime columns are in the correct format as per the raw data validation step.
    
    Args:
        raw_df (DataFrame): The input dataframe containing raw bikeshare trip data.

    Returns:
        DataFrame: The cleaned dataframe with standardized formats and no duplicates.
    """
    try:
        if not isinstance(raw_df, DataFrame):
            raise InvalidArgumentTypeError("Input raw_df must be a pandas DataFrame")
        
        raw_df = raw_df.drop_duplicates(subset=['ride_id'], keep='first').reset_index(drop=True)

        string_columns = [
            "ride_id", "rideable_type", "start_station_name", "start_station_id",
            "end_station_name", "end_station_id", "member_casual"
        ]
        
        for col in string_columns:
            raw_df[col] = raw_df[col].astype(str)
            raw_df[col] = raw_df[col].str.strip().str.lower()
        

        # print("cleaned_raw_df", raw_df)

        logger.info({
            "status": "success",
            "message": "Raw data cleaned successfully",
            "data_shape": raw_df.shape
        })
        return {
            "status": "success",
            "message": "Raw data cleaned successfully",
            "data": raw_df
        }
    
    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while cleaning raw data",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while cleaning raw data",
            "error": exc_error
        })



def standardize_coordinates_and_fill_station_ids(record: Dict, 
                                                 station_id_columns_dict: dict) -> Dict:
    """
    This function standardizes the latitude and longitude coordinates to 6 decimal places
    and fills missing station IDs using the provided data dictionary.

    Steps:
    1. standardize the latitude and longitude values to 6 decimal places as strings.
    2. Fill missing station IDs using the provided data dictionary for corresponding latitude and longitude.
    3. Standardize the station IDs(ensure the all the station IDs are hashed) and station names columns to lowercase and strip leading/trailing whitespace.

    Args:
        data (Dict): A dictionary containing the bikeshare trip data.

    Returns:
        Dict: A dictionary containing the status, message, and the updated data with standardized coordinates and filled station IDs.
    """
    try:
        if not isinstance(record, dict):
            raise InvalidArgumentTypeError("Input record must be a dictionary")
        if not isinstance(station_id_columns_dict, dict):
            raise InvalidArgumentTypeError("Input station_id_columns_dict must be a dictionary")
        
        cnt_loop = 0
        all_cleaned_records_per_batch = []
        station_id_columns = list(station_id_columns_dict.keys())
        store_modified_record = []
        final_modified_record = []

        for key, value in record.items():
            modified_record = {}
            if key in station_id_columns:
                coordinate_columns = station_id_columns_dict[key]['coordinate_columns']
                station_name_column = station_id_columns_dict[key]['station_name_column']

                if (record[coordinate_columns[0]] not in ['nan', float('nan')] \
                      and record[coordinate_columns[1]] not in ['nan', float('nan')]) and \
                        (pd.notna(record[coordinate_columns[0]]) and pd.notna(record[coordinate_columns[1]])):
                    
                    # Standardizing latitude and longitude to 6 decimal places as strings
                    latitude = f"{round(float(record[coordinate_columns[0]]), 6):.6f}"
                    longitude = f"{round(float(record[coordinate_columns[1]]), 6):.6f}"
                    modified_record[coordinate_columns[0]] = latitude
                    modified_record[coordinate_columns[1]] = longitude

                    # Filling missing station IDs
                    if latitude not in ['nan', float('nan')] and longitude not in ['nan', float('nan')]:

                        address_api_response = get_address(latitude, longitude)
                        # print("address_api_response", address_api_response)
                        if address_api_response['status'] == 'success':
                            print("record_station_name_column", record[station_name_column])
                            print("type_record_station_name", type(record[station_name_column]))
                            if record[station_name_column] in ['nan', float('nan')] or \
                                pd.isna(record[station_name_column]):
                                modified_record[station_name_column] = address_api_response['data']
                                print("fetched station name", modified_record[station_name_column])
                            else:
                                modified_record[station_name_column] = record[station_name_column]
                        else:
                            raise Exception(address_api_response)
                    else:
                        modified_record[station_name_column] = np.nan
                    
                    #fill station id for both empty and valid station id, in the case of valid station id, we ensure they all follow same hash256 format
                    if (latitude not in ['nan', float('nan')] and \
                        longitude not in ['nan', float('nan')]) and \
                            (pd.notna(latitude) and pd.notna(longitude)):
                        add_station_id_response = add_station_id(latitude, longitude)
                        # print("add_station_id_response", add_station_id_response)
                        if add_station_id_response['status'] == 'success':
                            modified_record[key] = add_station_id_response['data']
                        else:
                            raise Exception(add_station_id_response)
                    else:
                        modified_record[key] = np.nan
                    
                    # will add this as another function 
                    # Standardizing station name and station ID columns
                    if pd.isna(modified_record[station_name_column]) is False:
                        print("station name column before standardization", station_name_column)
                        print("station name before standardization", modified_record[station_name_column])
                        modified_record[station_name_column] = modified_record[station_name_column].strip().lower()
                    if pd.isna(modified_record[key]) is False:
                        modified_record[key] = modified_record[key].strip().lower()

                    # Retain other columns and their values in the record
                    modified_record['ride_id'] = record['ride_id']
                    modified_record['rideable_type'] = record['rideable_type']
                    modified_record['started_at'] = record['started_at']
                    modified_record['ended_at'] = record['ended_at']
                    modified_record['member_casual'] = record['member_casual']

                    store_modified_record.append(modified_record)

                    if cnt_loop > 0:
                        #merge end station and start station modified records
                        merged_modified_record = store_modified_record[cnt_loop] | store_modified_record[cnt_loop - 1]
                        # print("merged_modified_record", merged_modified_record)
                        final_modified_record.append(merged_modified_record)
                        # print("final_modified_record", final_modified_record)

                    cnt_loop += 1
                
                else:
                    modified_record[station_name_column] = np.nan
                    print("for nan entries")
                    print("station name set to nan", modified_record[station_name_column])

                    modified_record[key] = np.nan
                    modified_record.update({k: v for k, v in record.items() if k not in [coordinate_columns[0], coordinate_columns[1], station_name_column, station_id_columns[0], station_id_columns[1]]})
                    store_modified_record.append(modified_record)
                    if cnt_loop > 0:
                        #merge end station and start station modified records
                        merged_modified_record = store_modified_record[cnt_loop] | store_modified_record[cnt_loop - 1]
                        final_modified_record.append(merged_modified_record)
                    cnt_loop += 1

        if final_modified_record:
            all_cleaned_records_per_batch.append(final_modified_record[0])

        logger.info({
                        "status": "success",
                        "message": "Coordinates standardized and station IDs filled successfully",
                    })
        
        return {
            "status": "success",
            "message": "Coordinates standardized and station IDs filled successfully",
            "data": all_cleaned_records_per_batch
                }
    
    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while standardizing coordinates and filling station IDs",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while standardizing coordinates and filling station IDs",
            "error": exc_error
        })



def impute_missing_station_ids(df: DataFrame,
                                batch_size: int) -> DataFrame:
    """
    Process the bikeshare data to impute missing station IDs based on latitude and longitude using OpenStreetMap API.
    
    During data ingestion, some of the start_station_id and end_station_id are missing (about 13k entries in total), 
    and many of these entries have their corresponding latitude and longitude.
    This function uses the latitude and longitude to fetch the address from OpenStreetMap API and generate a station ID.

    Steps:
    1. Convert the DataFrame to a list of dictionaries for easier processing.
    2. Iterate through the records in batches to respect API rate limits.
    3. For each record, check if 'start_station_id' or 'end_station_id' is missing.
    4. If missing, use the corresponding latitude and longitude to fetch the address and generate a station ID.
    5. Handle exceptions and log errors appropriately.
    6. Combine all processed batches into a single DataFrame.

    
    Parameters:
    df (DataFrame): The input dataframe containing bikeshare trip data.
    batch_size (int): The number of records to process in each batch. 
    
    Returns:
    Dict: A dictionary containing the status, message, and the processed DataFrame with imputed station IDs.
    """
    try:
        
        if not isinstance(df, DataFrame):
            raise InvalidArgumentTypeError("Input df must be a pandas DataFrame")
        if not isinstance(batch_size, int) or (batch_size <= 0):
            print("batch_size", batch_size)
            raise InvalidArgumentTypeError("Input batch_size must be a positive integer")

        start_time = datetime.now()

        if df.empty:
            logger.info({
                "status": "success",
                "message": "Input dataframe is empty, no processing of raw data needed.",
                "data_shape": df.shape
            })
            return {
                "status": "success",
                "message": "Input dataframe is empty, no processing of raw data needed.",
                "data": df
            }

        df_dict = df.to_dict(orient='records')

        all_clean_merged_records = []
        station_id_columns_dict = {
            'start_station_id': {
                "coordinate_columns": ('start_lat', 'start_lng'),
                "station_name_column": 'start_station_name'
            },
            'end_station_id': {
                "coordinate_columns": ('end_lat', 'end_lng'),
                "station_name_column": 'end_station_name'
            }
        }
        batch_count = 0
        for i in range(0, len(df_dict), batch_size):
            batch = df_dict[i:i + batch_size]

            batch_records = []
            for record in batch:
                transform_status = standardize_coordinates_and_fill_station_ids(record, station_id_columns_dict)
                # print("transform_status", transform_status)
                if transform_status['status'] == 'success':
                    cleaned_record = transform_status['data']
                    batch_records.append(cleaned_record[0])
                else:
                    raise Exception(transform_status)
            
            batch_count += 1              
            # logger.info({
            #     "status": "success",
            #     "message": f"Successfully extracted station address and processed station ID for batch: {batch_count}",
            # })
            all_clean_merged_records.append(pd.DataFrame(batch_records))

        transformed_df = pd.concat(all_clean_merged_records, ignore_index=True)
        #arrange columns
        transformed_df = transformed_df[[
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]]

        end_time = datetime.now()
        time_taken = end_time - start_time

        logger.info({
            "status": "success",
            "message": f"Successfully processed bikeshare data with imputed station IDs. Time taken: {time_taken}"
        })
        
        return {
            "status": "success",
            "message": f"Successfully processed bikeshare data with imputed station IDs.",
            "data": transformed_df
        }
    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while imputing station IDs in bikeshare data",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while imputing station IDs in bikeshare data",
            "error": exc_error
        })



def validate_processed_data(s3_config: Dict,
                            processed_data_key: str) -> Dict:
    """
    This function validates the processed bikeshare data to ensure it meets the required schema and data quality standards.
    
    Args:
        s3_key (str): The S3 key for the processed data.

    Returns:
        Dict: A dictionary containing the status and message of the validation operation.
    """

    try:
        if not isinstance(s3_config, dict):
            raise InvalidArgumentTypeError("Input s3_config must be a dictionary")
        if not isinstance(processed_data_key, str):
            raise InvalidArgumentTypeError("Input processed_data_key must be a string")
        
        logger.info({
            "status": "info",
            "message": "Starting processed data validation",
            "processed_data_key": processed_data_key
        })

        s3_client = boto3.client(
            's3',
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key']
        )
        transformed_bucket = s3_config['transformed_bucket']

        # Download the staging data from S3
        staging_data_obj = s3_client.get_object(Bucket=transformed_bucket, Key=processed_data_key)
        # print("staging_data_obj", staging_data_obj)

        body_reader = staging_data_obj['Body']
        # print("body_reader", body_reader)

        staging_df = pd.read_parquet(io.BytesIO(staging_data_obj['Body'].read()))
        # print("staging_df", staging_df.info())

        ge_df_staging = ge.from_pandas(staging_df)
        required_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]

        validate_column_existence = ge_df_staging.expect_table_columns_to_match_set(required_cols)
        if not validate_column_existence['success']:
            missing_cols = set(required_cols) - set(ge_df_staging.get_table_columns())
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": f"columns: {sorted(missing_cols)} are missing"
            })
        
        non_null_columns = [
        "ride_id", "rideable_type", "started_at",
        "ended_at", "start_station_id",
            "start_station_name", "start_lat", "start_lng",
            "member_casual"]

       
        validate_non_null_col = ge_df_staging.expect_column_values_to_not_be_null(non_null_columns)
        if validate_non_null_col['success'] is False:
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": "At least one non_null column contains nulls"
            })
        
        validate_ride_id_unique = ge_df_staging.expect_column_values_to_be_unique("ride_id")
        if validate_ride_id_unique['success'] is False:
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": "ride_id column contains duplicate values"
            })
        
        date_time_cols = ["started_at", "ended_at"]
        invalid_datetime_cols = []
        for col in date_time_cols:
            validate_datetime_format = ge_df_staging.expect_column_values_to_match_strftime_format(col, "%Y-%m-%d %H:%M:%S")
            if validate_datetime_format['success'] is False:
                invalid_datetime_cols.append(col)
        if invalid_datetime_cols:
            raise Exception({
                    "status": "error",
                    "message": "column validation error occured",
                    "error": f"{invalid_datetime_cols} contains invalid datetime format"
                })
        
        str_cols = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_name",
                     "start_station_id", "end_station_name", 
                     "end_station_id", "member_casual"]
    
        invalid_str_cols = []

        for col in str_cols:
            validate_str_type = ge_df_staging.expect_column_values_to_be_of_type(col, "str")
            if validate_str_type['success'] is False:
                invalid_str_cols.append(col)

        if invalid_str_cols:
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": f"{invalid_str_cols} contains non-string values"
            }) 

        validate_ride_id_length = ge_df_staging.expect_column_value_lengths_to_equal('ride_id', 16)
        if validate_ride_id_length['success'] is False:
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": "ride_id column contains values with incorrect length"
            })

        validate_start_station_id_length = ge_df_staging.expect_column_value_lengths_to_equal('start_station_id', 64)
        if validate_start_station_id_length['success'] is False:
            raise Exception({
                "status": "error",
                "message": "column validation error occured",
                "error": "start_station_id column contains values with incorrect length"
            })

        logger.info({
            "status": "success",
            "message": "Processed data validation completed successfully",
            "data_shape": ge_df_staging.shape
        })
        return {
            "status": "success",
            "message": "Processed data validation completed successfully",
            "data_shape": ge_df_staging.shape
        }
    
    except Exception as e:
        try:
            gexp_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            gexp_error = str(e)

        error_logger.error({
            "status": "error",
            "message": "An error occurred while validating processed data",
            "error": gexp_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while validating processed data",
            "error": gexp_error
        })

def list_partitions(bucket: str,
                    raw_bucket_prefix: str,
                    access_key: str,
                    secret_key: str) -> dict[tuple, list[str]]:
    """
    This function lists the partitions in the raw S3 bucket based on the specified prefix and returns a dictionary keyed by (year, week) with sorted ingestion_ts paths.

    Args:
        bucket (str): The name of the S3 bucket.
        raw_bucket_prefix (str): The prefix/folder in the S3 bucket where the data is stored.
        access_key (str): The AWS access key for authentication.
        secret_key (str): The AWS secret key for authentication.

    Returns:
        dict: A dictionary containing the status, message, and a nested dictionary keyed by (year, week) with sorted ingestion timestamp paths.

        Example of returned partitions structure:
        partitions = {
            ('2022', '48'): [
                ('2026-02-03T20-11-44Z',
                'trips/year=2022/week=48/day=Mon/ingestion_ts=2026-02-03T20-11-44Z/trips.csv'),
                ('2026-02-04T09-10-00Z',
                'trips/year=2022/week=48/day=Mon/ingestion_ts=2026-02-04T09-10-00Z/trips.csv')
            ]
        }

    """
    try:
        if not isinstance(bucket, str):
            raise InvalidArgumentTypeError("bucket argument must be a string")
        if not isinstance(raw_bucket_prefix, str):
            raise InvalidArgumentTypeError("raw_bucket_prefix argument must be a string")
        if not isinstance(access_key, str):
            raise InvalidArgumentTypeError("access_key argument must be a string")
        if not isinstance(secret_key, str):
            raise InvalidArgumentTypeError("secret_key argument must be a string")
        
        s3_client = boto3.client("s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key)

        print("s3_client", s3_client)
        
        paginator = s3_client.get_paginator("list_objects_v2")
        
        partitions: dict[tuple, list[str]] = {}
        
        for page in paginator.paginate(Bucket=bucket, Prefix=raw_bucket_prefix):
            print("page", page)
            for obj in page.get("Contents", []):
                key = obj["Key"]
                print('key', key)

                key_match = re.search(
                    r"year=(\d+)/week=(\d+)/ingestion_ts=([\w\-:T]+Z)/",
                    key
                )
                if key_match and key.endswith(".csv"):
                    year, week, ts = key_match.group(1), key_match.group(2), key_match.group(3)
                    partition_key = (year, week)
                    partitions.setdefault(partition_key, [])
                    partitions[partition_key].append((ts, key))

        for year_week_key in partitions:
            partitions[year_week_key].sort(key=lambda x: x[0])
        
        logger.info({
            "status": "success",
            "message": f"Successfully listed partitions in bucket: {bucket} with folder: {raw_bucket_prefix}",
            "partitions": partitions
        })

        return {
            "status": "success",
            "message": f"Successfully listed partitions in bucket: {bucket} with folder: {raw_bucket_prefix}",
            "partitions": partitions
        }
    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": f"An error occurred while listing partitions in bucket: {bucket} with folder: {raw_bucket_prefix}",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": f"An error occurred while listing partitions in bucket: {bucket} with folder: {raw_bucket_prefix}",
            "error": exc_error
        })


def yield_event_stream(raw_bucket: str,
                            raw_bucket_prefix: str,
                            aws_access_key: str,
                            aws_secret_key: str,
                            chunk_size: str,
                            delay_seconds: int) -> Generator[Dict, None, None]:
    """
    This function reads the latest data for each (year, week) partition from the raw S3 bucket and yields each row as a dictionary with additional metadata.
    For each (year, week), reads ONLY the latest ingestion run.
    This is an idempotency guard against reruns.

    Args:
        raw_bucket (str): The name of the raw S3 bucket.
        raw_bucket_prefix (str): The prefix/folder in the raw S3 bucket where the
                            data is stored.
        aws_access_key (str): The AWS access key for authentication.
        aws_secret_key (str): The AWS secret key for authentication.
        chunk_size (int): The number of lines to read at once from the CSV file.
        delay_seconds (int): The delay in seconds between yielding each line to simulate real-time data streaming.
    
    Yields:
        dict: A dictionary representing a single row from the CSV file, enriched with '_year', '_week', and '_ingestion_ts' metadata.
    """
    try:
        if not isinstance(raw_bucket, str):
            raise InvalidArgumentTypeError("raw_bucket argument must be a string")

        if not isinstance(raw_bucket_prefix, str):
            raise InvalidArgumentTypeError("raw_bucket_prefix argument must be a string")
        
        if not isinstance(aws_access_key, str):
            raise InvalidArgumentTypeError("aws_access_key argument must be a string")
        
        if not isinstance(aws_secret_key, str):
            raise InvalidArgumentTypeError("aws_secret_key argument must be a string")

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise InvalidArgumentTypeError("chunk_size argument must be a positive integer")
        
        if not isinstance(delay_seconds, int) or delay_seconds < 0:
            raise InvalidArgumentTypeError("delay_seconds argument must be a non-negative integer")
        
        s3_client = boto3.client("s3",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key)
        list_partitions_status = list_partitions(raw_bucket, raw_bucket_prefix, aws_access_key, aws_secret_key)
        partitions = list_partitions_status["partitions"]

        for (year, week), ts_key_pairs in partitions.items():
            latest_ts, latest_key = ts_key_pairs[-1]  
            
            obj = s3_client.get_object(Bucket=raw_bucket, Key=latest_key)
            body = obj["Body"].read().decode("utf-8")
            
            for chunk in pd.read_csv(StringIO(body), chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    event = row.to_dict()
                    event['_year'] = year
                    event['_week'] = week
                    event['_ingestion_ts'] = latest_ts
                    yield event
                    time.sleep(delay_seconds)

    except Exception as e:
        try:
            exc_error = ast.literal_eval(str(e))
        except SyntaxError as se:
            exc_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while yielding event stream from S3",
            "error": exc_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while yielding event stream from S3",
            "error": exc_error

        })
    
def send_slack_alert(
        message: str,
        channel: str,
        oauth_token: str,
        slack_bot: str
) -> Dict[str, str]:
    """
    Sends an alert message to a specified Slack channel using the Slack WebClient.

    Args:
        message (str): The message to be sent to the Slack channel.
        channel (str): The Slack channel ID where the message will be sent.
        oauth_token (str): The OAuth token for authenticating with the Slack API.
        slack_bot (str): The name of the Slack bot sending the message.

    Returns:
        Dict[str, str]: A dictionary containing the status and message of the operation.
    """
    try:
        if not isinstance(message, str):
            raise InvalidArgumentTypeError(
                "message argument must be a string"
            )
        if not isinstance(channel, str):
            raise InvalidArgumentTypeError(
                "channel argument must be a string"
            )
        if not isinstance(oauth_token, str):
            raise InvalidArgumentTypeError("oauth_token argument must be a string")
        
        if not isinstance(slack_bot, str):
            raise InvalidArgumentTypeError("slack_bot argument must be a string")
        
        slack_client = WebClient(token=oauth_token)
        response = slack_client.chat_postMessage(
            channel=channel,
            text=message,
            username=slack_bot
        )

        if response['ok']:
            logger.info({
                "status": "success",
                "message": f"Alert sent to Slack: {channel} successfully"
            })
            return {
                "status": "success",
                "message": f"Alert sent to Slack: {channel} successfully"
            }

    except Exception as e:
        try:
            gexp_error = ast.literal_eval(str(e))
        except SyntaxError as se:
            gexp_error = str(e)

        error_logger.error({
            "status": "error",
            "message": "An error occurred while sending alert to Slack",
            "error": gexp_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while sending alert to Slack",
            "error": gexp_error
        })

def write_flagged_event_to_snowflake(event: Dict,
                                    snowflake_config: Dict) -> Dict:
    """
    Writes a flagged event to a Snowflake table.

    Args:
        event (Dict): The flagged event data to be written.
        snowflake_config (Dict): The Snowflake configuration parameters.

    Returns:
        Dict: A dictionary containing the status and message of the operation.
    """
    try:
        if not isinstance(event, dict):
            raise InvalidArgumentTypeError("event argument must be a dictionary")
        if not isinstance(snowflake_config, dict):
            raise InvalidArgumentTypeError("snowflake_config argument must be a dictionary")
        
        required_snowflake_config_keys = [
            "snowflake_username",
            "snowflake_password",
            "snowflake_account",
            "snowflake_database",
            "snowflake_schema",
            "snowflake_warehouse",
            "snowflake_role"
        ]

        for key in required_snowflake_config_keys:
            if key not in snowflake_config:
                raise KeyError(f"missing required snowflake config key: {key}")
        
        required_event_keys = [
            "id",
            "ride_id",
            "flag_type",
            "rideable_type",
            "started_at",
            "ended_at",
            "start_station_name",
            "start_station_id",
            "end_station_name",
            "end_station_id",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
            "member_casual"
        ]

        for key in required_event_keys:
            if key not in event:
                raise KeyError(f"missing required event key: {key}")


        user = snowflake_config['snowflake_username']
        password = snowflake_config['snowflake_password']
        account = snowflake_config['snowflake_account']
        database = snowflake_config['snowflake_database']
        schema = snowflake_config['snowflake_schema']
        warehouse = snowflake_config['snowflake_warehouse']
        role = snowflake_config['snowflake_role']

        conn_string = f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}"
        alerts_schema = AlertsLog.__table_args__['schema']
        table_name = AlertsLog.__tablename__

        with create_session(conn_string) as db:

                sql_query = text(f"""
                    INSERT INTO {database}.{alerts_schema}.{table_name} (
                        id,
                        ride_id,
                        flag_type,
                        rideable_type,
                        started_at,
                        ended_at,
                        start_station_name,
                        start_station_id,
                        end_station_name,
                        end_station_id,
                        start_lat,
                        start_lng,
                        end_lat,
                        end_lng,
                        member_casual,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :id,
                        :ride_id,
                        :flag_type,
                        :rideable_type,
                        :started_at,
                        :ended_at,
                        :start_station_name,
                        :start_station_id,
                        :end_station_name,
                        :end_station_id,
                        :start_lat,
                        :start_lng,
                        :end_lat,
                        :end_lng,
                        :member_casual,
                        CURRENT_TIMESTAMP()::timestamp_ntz,
                        CURRENT_TIMESTAMP()::timestamp_ntz
                    )
                        """
                        )
                            
                execute_query = db.execute(sql_query,
                    {
                        'id': event['id'],
                        'ride_id': event['ride_id'],
                        'flag_type': event['flag_type'],
                        'rideable_type': event['rideable_type'],
                        'started_at': event['started_at'],
                        'ended_at': event['ended_at'],
                        'start_station_name': event['start_station_name'],
                        'start_station_id': event['start_station_id'],
                        'end_station_name': event['end_station_name'],
                        'end_station_id': event['end_station_id'],
                        'start_lat': event['start_lat'],
                        'start_lng': event['start_lng'],
                        'end_lat': event['end_lat'],
                        'end_lng': event['end_lng'],
                        'member_casual': event['member_casual']
                    }
                                        )
                db.commit()

        logger.info({
            "status": "success",
            "message": "Flagged event written to Snowflake successfully"
        })
        return {
            "status": "success",
            "message": "Flagged event written to Snowflake successfully"
        }
        
    except Exception as e:
        try:
            f_error = ast.literal_eval(str(e))
        except SyntaxError as se:
            f_error = str(e)
        error_logger.error({
            "status": "error",
            "message": "An error occurred while writing flagged event to Snowflake",
            "error": f_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while writing flagged event to Snowflake",
            "error": f_error
        })




