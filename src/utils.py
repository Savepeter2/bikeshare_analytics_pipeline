import hashlib
import os
import random
import string
import sys
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Dict, List, Union

from configs.logger_config import error_logger, logger
import requests
import time
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict
import boto3
from typing import Set, Dict
from io import BytesIO
import io
import great_expectations as ge
from requests.sessions import Session
from src.schema import raw_data_schema



def gen_hash_key_station_id(latitude: str, longitude: str) -> Dict[str, str]:
    """
    Function to generate a hash key for a given station, either start or end station
    The hash key is generated using the latitude and longitude of the station
    
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

            # logger.info(
            #     {
            #         "status": "success",
            #         "message": "Hash key generated successfully",
            #         "hash_key": hash_key,
            #     }
            # )

            return {
                "status": "success",
                "message": "Hash key generated successfully for station ID",
                "hash_key": hash_key,
            }

        else:
            raise ValueError("Latitude and Longitude for generating station ID must be string types")

    except Exception as e:
        error_logger.error(
            {
                "status": "error",
                "message": "Unable to generate hash key for station ID",
                "error": str(e),
            }
        )
        raise Exception(
            {
                "status": "error",
                "message": "Unable to generate hash key for station ID",
                "error": str(e),
            }
        )

def extract_source_data(
    aws_access_key: str,
    aws_secret_key: str,
    source_bucket: str,
    source_s3_key: str,
    batch_size: int,
    last_row_index: int,
    raw_data_schema: Dict[str, str]
) -> Dict:
    """
    Extracts data from the source S3 bucket in batches. 

    Args:
        - aws_access_key (str): AWS access key for S3 authentication.
        - aws_secret_key (str): AWS secret key for S3 authentication.
        - source_bucket (str): Name of the source S3 bucket.
        - source_s3_key (str): Key of the source S3 object.
        - batch_size (int): Number of rows to extract in each batch.
        - last_row_index (int): The last row index that was extracted in the previous batch.
    Returns:
        - Dict: A dictionary containing the status, message, and the extracted DataFrame.
    """
    try:
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

        start_index = last_row_index + 1
        end_index = start_index + batch_size - 1

        raw_df = pd.read_csv(source_data,
                            skiprows=range(1, start_index + 1),
                            nrows=batch_size,
                            dtype=raw_data_schema    
                             )
 
        if raw_df.empty:
            raise Exception({
                "status": "error",
                "message": f"No data found in the source s3 path for rows: {start_index} to {end_index}",
                "data_shape": raw_df.shape
            })
        
        logger.info({
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}, rows: {start_index} to {end_index}",
            "data_shape": raw_df.shape
        })
        return {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}, rows: {start_index} to {end_index}",
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
        ge_df_raw = ge.from_pandas(raw_df)
        required_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]

        for col in required_cols:
            validate_column_existence = ge_df_raw.expect_column_to_exist(col)
            if validate_column_existence['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": "Column Validation error occured",
                    "error": f"Column name: '{col}' is not expected"
                })
        
        validate_ride_id_not_null = ge_df_raw.expect_column_values_to_not_be_null("ride_id")
        if validate_ride_id_not_null['success'] is False:
            raise Exception({
                "status": "error",
                "message": "ride_id contains null values",
                "error": validate_ride_id_not_null
            })
        
        validate_ride_id_unique = ge_df_raw.expect_column_values_to_be_unique("ride_id")
        if validate_ride_id_unique['success'] is False:
            raise Exception({
                "status": "error",
                "message": "ride_id contains duplicate values",
                "error": validate_ride_id_unique
            })
        
        date_time_cols = ["started_at", "ended_at"]
        for col in date_time_cols:
            validate_datetime_format = ge_df_raw.expect_column_values_to_match_strftime_format(col, "%Y-%m-%d %H:%M:%S")
            if validate_datetime_format['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains invalid datetime format",
                    "error": validate_datetime_format
                })
        
        str_cols = ["ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "end_station_name", "member_casual" ]
        for col in str_cols:
            validate_str_type = ge_df_raw.expect_column_values_to_be_of_type(col, "str")
            if validate_str_type['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains non-string values",
                    "error": validate_str_type
                }) 
        
        float_cols = [
            "start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng"
        ]
        for col in float_cols:
            validate_float_type = ge_df_raw.expect_column_values_to_be_of_type(col, "float")
            if validate_float_type['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains non-float values",
                    "error": validate_float_type
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
        error_logger.error({
            "status": "error",
            "message": "An error occurred during raw data validation",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred during raw data validation",
            "error": str(e)
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
            raise ValueError("Latitude and Longitude must be string types")
        
        headers = {
                "User-Agent": "BikeshareLocationService/1.0 (contact: saveadekolu@gmail.com)",
                "Accept-Language": "en"
        }
        
        session = requests_session_with_retries()
        osm_api_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        response = session.get(osm_api_url, headers=headers)
        status_code = response.status_code
        
        if status_code == 200:
            data = response.json()
            address = data.get('display_name', 'Address not found')
            # logger.info({
            #     "status": "success",
            #     "message": f"Address fetched from OpenStreetMap API successfully for lat: {lat}, lon: {lon}",
            # })
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
                "message": f"Failed to fetch data from OpenStreetMap API. Status code: {status_code}",
                "error": response.text
            })
            raise Exception({
                "status": "error",
                "message": f"Failed to fetch data from OpenStreetMap API. Status code: {status_code}",
                "error": response.text
            })

    except ConnectionError as ce:
        error_logger.error({"status": "error",
                            "message": "ConnectionError occurred while fetching address from OpenStreetMap API, retrying in 30 seconds...",
                            "error": {str(ce)}})
        time.sleep(30)
        retry_api_response = get_address(lat, lon)
        return retry_api_response
    
    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while fetching address from OpenStreetMap API",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while fetching address from OpenStreetMap API",
            "error": str(e)
        })


def add_station_id(lat: str, lon: str) -> Dict:
    """
    This function adds an ID for both start station ID and end station ID, marking their identifiers

    Args:

    lat(str): Latitude of the station 
    lon(str): Longitude of the station 

    Returns: 



    """
    try:
        if not isinstance(lat, str) and not isinstance(lon, str):
            raise ValueError("Latitude and Longitude must be string types")

        response = gen_hash_key_station_id(lat, lon)
        station_id = response['hash_key']

        return {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": station_id
        }


    except Exception as e:
        error_logger.error(f"Exception occurred in add_station_id: {str(e)}")
        raise Exception({
            "status": "error",
            "message": "An error occurred while generating station ID",
            "error": str(e)
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
            raise ValueError("Input raw_df must be a pandas DataFrame")
        
        raw_df = raw_df.drop_duplicates(subset=['ride_id'], keep='first').reset_index(drop=True)

        string_columns = [
            "ride_id", "rideable_type", "start_station_name", "start_station_id",
            "end_station_name", "end_station_id", "member_casual"
        ]
        
        for col in string_columns:
            raw_df[col] = raw_df[col].astype(str)
            raw_df[col] = raw_df[col].str.strip().str.lower()

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
        error_logger.error({
            "status": "error",
            "message": "An error occurred while cleaning raw data",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while cleaning raw data",
            "error": str(e)
        })

def standardize_coordinates_and_fill_station_ids(record: Dict, 
                                                 station_id_columns_dict: dict) -> Dict:
    """
    This function standardizes the latitude and longitude coordinates to 6 decimal places
    and fills missing station IDs using the provided data dictionary.

    Steps:
    1. standardize the latitude and longitude values to 6 decimal places as strings.
    2. Fill missing station IDs using the provided data dictionary for corresponding latitude and longitude.
    3. Standardize the station IDs and station names columns to lowercase and strip leading/trailing whitespace.

    Args:
        data (Dict): A dictionary containing the bikeshare trip data.

    Returns:
        Dict: A dictionary containing the status, message, and the updated data with standardized coordinates and filled station IDs.
    """
    try:
        if not isinstance(record, dict):
            raise ValueError("Input record must be a dictionary")
        if not isinstance(station_id_columns_dict, dict):
            raise ValueError("Input station_id_columns_dict must be a dictionary")
        
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

                if record[coordinate_columns[0]] != 'nan' and record[coordinate_columns[1]] != 'nan':

                    # Standardizing latitude and longitude to 6 decimal places as strings
                    latitude = f"{round(float(record[coordinate_columns[0]]), 6):.6f}"
                    longitude = f"{round(float(record[coordinate_columns[1]]), 6):.6f}"
                    modified_record[coordinate_columns[0]] = latitude
                    modified_record[coordinate_columns[1]] = longitude

                    # Filling missing station IDs
                    if latitude != 'nan' and longitude != 'nan':
                        address_api_response = get_address(latitude, longitude)
                        if address_api_response['status'] == 'success':
                            if record[station_name_column] == 'nan':
                                modified_record[station_name_column] = address_api_response['data']
                                print("fetched station name", modified_record[station_name_column])
                            else:
                                modified_record[station_name_column] = record[station_name_column]
                        else:
                            raise Exception(address_api_response)
                    else:
                        modified_record[station_name_column] = str(np.nan)
                    
                    #fill station id for both empty and valid station id, in case of valid station id, we ensure they all follow same hash256 format
                    if latitude != 'nan' and longitude != 'nan':
                        add_station_id_response = add_station_id(latitude, longitude)
                        if add_station_id_response['status'] == 'success':
                            modified_record[key] = add_station_id_response['data']
                        else:
                            raise Exception(add_station_id_response)
                    else:
                        modified_record[key] = str(np.nan)
                    

                    # Standardizing station name and station ID columns
                    if modified_record[station_name_column] != 'nan':

                        modified_record[station_name_column] = modified_record[station_name_column].strip().lower()
                    if modified_record[key] != 'nan':
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
                        final_modified_record.append(merged_modified_record)

                    cnt_loop += 1
                
                else:
                    modified_record[station_name_column] = str(np.nan)
                    modified_record[key] = str(np.nan)
                    modified_record.update({k: v for k, v in record.items() if k not in [coordinate_columns[0], coordinate_columns[1], station_name_column, station_id_columns[0], station_id_columns[1]]})
                    store_modified_record.append(modified_record)
                    if cnt_loop > 0:
                        #merge end station and start station modified records
                        merged_modified_record = store_modified_record[cnt_loop] | store_modified_record[cnt_loop - 1]
                        final_modified_record.append(merged_modified_record)
                    cnt_loop += 1

        if final_modified_record:
            all_cleaned_records_per_batch.append(final_modified_record[0])
        

        # logger.info({
        #                 "status": "success",
        #                 "message": "Coordinates standardized and station IDs filled successfully",
        #             })
        return {
            "status": "success",
            "message": "Coordinates standardized and station IDs filled successfully",
            "data": all_cleaned_records_per_batch
    }
    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while standardizing coordinates and filling station IDs",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while standardizing coordinates and filling station IDs",
            "error": str(e)
        })



def impute_missing_station_ids(df: DataFrame, batch_size: int) -> DataFrame:
    """
    Process the bikeshare data to impute missing station IDs based on latitude and longitude using OpenStreetMap API.
    
    During data ingestion, some of the start_station_id and end_station_id are missing (about 13k entries in total), 
    and may of these entries have their corresponding latitude and longitude.
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
        start_time = datetime.now()
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
                if transform_status['status'] == 'success':
                    cleaned_record = transform_status['data']
                    batch_records.append(cleaned_record[0])
                else:
                    raise Exception(transform_status)
            
            batch_count += 1              
            logger.info({
                "status": "success",
                "message": f"Successfully extracted station address and processed station ID for batch: {batch_count}",
            })
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
            "message": f"Successfully processed bikeshare data with imputed station IDs. Time taken: {time_taken}",
            "data": transformed_df

        }
    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while imputing station IDs in bikeshare data",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while imputing station IDs in bikeshare data",
            "error": str(e)
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
            raise ValueError("Input s3_config must be a dictionary")
        if not isinstance(processed_data_key, str):
            raise ValueError("Input processed_data_key must be a string")
        
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
        staging_df = pd.read_parquet(io.BytesIO(staging_data_obj['Body'].read()))

        print("staging_df", staging_df.info())

        ge_df_staging = ge.from_pandas(staging_df)
        required_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]

        for col in required_cols:
            validate_column_existence = ge_df_staging.expect_column_to_exist(col)
            if validate_column_existence['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": "Column Validation error occured",
                    "error": f"Column name: '{col}' is not expected"
                })
        
        not_null_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at", "member_casual"
        ]

        for col in not_null_cols:
            validate_not_null = ge_df_staging.expect_column_values_to_not_be_null(col)
            if validate_not_null['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains null values",
                    "error": validate_not_null
                })
        
        validate_ride_id_unique = ge_df_staging.expect_column_values_to_be_unique("ride_id")
        if validate_ride_id_unique['success'] is False:
            raise Exception({
                "status": "error",
                "message": "ride_id contains duplicate values",
                "error": validate_ride_id_unique
            })
        
        date_time_cols = ["started_at", "ended_at"]
        for col in date_time_cols:
            validate_datetime_format = ge_df_staging.expect_column_values_to_match_strftime_format(col, "%Y-%m-%d %H:%M:%S")
            if validate_datetime_format['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains invalid datetime format",
                    "error": validate_datetime_format
                })
        
        str_cols = ["ride_id", "rideable_type", "start_station_name",
                     "start_station_id", "end_station_name", 
                     "end_station_id", "member_casual", "start_lat", 
                     "start_lng", "end_lat", "end_lng"]
        
        for col in str_cols:
            validate_str_type = ge_df_staging.expect_column_values_to_be_of_type(col, "str")
            if validate_str_type['success'] is False:
                raise Exception({
                    "status": "error",
                    "message": f"{col} contains non-string values",
                    "error": validate_str_type
                }) 

        validate_ride_id_length = ge_df_staging.expect_column_value_lengths_to_equal('ride_id', 16)
        if validate_ride_id_length['success'] is False:
            raise Exception({
                "status": "error",
                "message": "ride_id contains invalid length value",
                "error": validate_ride_id_length
            })

        validate_start_station_id_length = ge_df_staging.expect_column_value_lengths_to_equal('start_station_id', 64)
        if validate_start_station_id_length['success'] is False:
            raise Exception({
                "status": "error",
                "message": "start_station_id contains invalid length value",
                "error": validate_start_station_id_length
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
        error_logger.error({
            "status": "error",
            "message": "An error occurred during processed data validation",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred during processed data validation",
            "error": str(e)
        })

