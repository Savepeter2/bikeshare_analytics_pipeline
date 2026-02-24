import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from src.etl import (extract_and_validate_source_data,
                     LoadToRawS3,
                        LoadToTransformedS3,
                        LoadTransformedDataToSnowflake)
from re import Match
from tests.test_utils import (moto_boto3_client, mock_boto3_client, aws_credentials, 
                                create_transformed_bucket, upload_transformed_data_to_s3,
                                valid_snowflake_config)
from pytest import fixture
import pandas as pd
from moto import mock_aws
from typing import Dict
from unittest.mock import patch, MagicMock
from botocore.client import BaseClient
import numpy as np
import boto3
import re
import io
from io import BytesIO
from builtins import bytes
from typing import Generator, Dict, Callable
from datetime import datetime
from botocore.exceptions import ClientError
from requests.sessions import Session


@pytest.fixture(scope="class")
def loadertosnowflake() -> None:
    """
    This fixture initializes an instance of LoadTransformedDataToSnowflake class

    Args:
        None

    Returns:
        An instance of LoadTransformedDataToSnowflake class.
    """
    loader = MagicMock(spec =LoadTransformedDataToSnowflake)
    loader.connection_string = "snowflake://test_connection_string"
    loader.database = "test_database"
    loader.schema = "test_schema"
    loader.table_name = "test_table"
    loader.stage_name = "test_stage"
    return loader

@pytest.fixture(scope="class")
def mock_bikeride() -> None:
    """
    This fixture provides a mocked bikeride table. 
    This mocked table serves as the table where data will be loaded to before being inserted into Snowflake.

    Args:
        None

    Returns:
        A mocked bikeride table.
    """
    class MockBikeRide:
        __tablename__ = "raw_bike_rides"
        __table_args__ = {'schema': 'raw',
                          'comment': 'Mocked bike rides table from transformed S3 bucket'}

    return MockBikeRide



@pytest.fixture(scope="class")
def valid_sample_raw_df() -> pd.DataFrame:
    """
    This fixture provides a valid sample raw DataFrame that mimics the structure of the raw data extracted from the source S3 bucket.

    Args:
        None

    Returns:
        A valid sample raw DataFrame.
    """

    sample_raw_df = pd.DataFrame({
                "ride_id": ["ride123", "ride124", "ride125"],
                "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
                "start_station_id": [100.0, 101.0, 102.0],
                "end_station_name": ["end_station_1", np.nan, np.nan],
                "end_station_id": [200.0, np.nan, np.nan],
                "start_lat": [40.7128, 34.0522, 41.8781],
                "start_lng": [-74.0060, -118.2437, -87.6298],
                "end_lat": [40.7589, np.nan, np.nan],
                "end_lng": [-73.9851, np.nan, np.nan],
                "member_casual": ["member", "member", "casual"],
            })
    
    return sample_raw_df

@pytest.fixture(scope="class")
def create_source_bucket(moto_boto3_client: BaseClient,
                         aws_credentials: dict) -> None:
    """
    Fixture to create a mock S3 source bucket.

    Args:
        moto_boto3_client: A mocked boto3 S3 client.
        aws_credentials: Mock AWS credentials.

    Returns:
        The mocked boto3 S3 client with the source bucket created.
    """
    source_bucket = aws_credentials['source_bucket']
    moto_boto3_client.create_bucket(Bucket=source_bucket)

@pytest.fixture(scope="class")
def upload_to_source_bucket(moto_boto3_client: BaseClient,
                            aws_credentials: dict) -> None:
    """
    Fixture to upload a sample CSV file to the mock S3 source bucket.

    Args:
        create_source_bucket: A mocked boto3 S3 client with the source bucket created.
        aws_credentials: Mock AWS credentials.

    Returns:
        None
    """
    source_bucket = aws_credentials['source_bucket']
    source_s3_key = aws_credentials['source_s3_key']

    sample_data = pd.DataFrame({
                "ride_id": ["ride123", "ride124", "ride125"],
                "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
                "start_station_id": [100.0, 101.0, 102.0],
                "end_station_name": ["end_station_1", np.nan, np.nan],
                "end_station_id": [200.0, np.nan, np.nan],
                "start_lat": [40.7128, 34.0522, 41.8781],
                "start_lng": [-74.0060, -118.2437, -87.6298],
                "end_lat": [40.7589, np.nan, np.nan],
                "end_lng": [-73.9851, np.nan, np.nan],
                "member_casual": ["member", "member", "casual"],
            })
 
    def upload_data(df: pd.DataFrame = sample_data) -> None:
        """
        Uploads sample data to the source bucket.

        Args:
            df: DataFrame containing the sample data to upload.

        Returns:
            None
        """
        df_csv = df.to_csv(index=False)

        moto_boto3_client.put_object(Bucket=source_bucket,
                                        Key=source_s3_key,
                                         Body=df_csv)
        print(f"Uploaded sample data to source bucket: {source_bucket}, s3 key: {source_s3_key}")
    return upload_data

@pytest.fixture(scope="class")
def create_raw_bucket(moto_boto3_client: BaseClient,
                      aws_credentials: dict) -> None:
    """
    Fixture to create a mock S3 raw bucket.

    Args:
        moto_boto3_client: A mocked boto3 S3 client.
        aws_credentials: Mock AWS credentials.

    Returns:
        None
    """
    raw_bucket = aws_credentials['raw_bucket']
    moto_boto3_client.create_bucket(Bucket=raw_bucket)

@pytest.fixture(scope="class")
def raw_data_schema() -> Dict[str, str]:
    """
    Fixture to provide the raw data schema.

    Returns:
        A dictionary representing the raw data schema.
    """

    raw_data_schema = {
                "ride_id": "object",
                "rideable_type": "object",
                "started_at": "object",
                "ended_at": "object",
                "start_station_name": "object",
                "start_station_id": "float64",
                "end_station_name": "object",
                "end_station_id": "float64",
                "start_lat": "float64",
                "start_lng": "float64",
                "end_lat": "float64",
                "end_lng": "float64",
                "member_casual": "object"
            }
    return raw_data_schema

@pytest.fixture(scope="class")
def extract_and_valid_source_params(raw_data_schema: Dict) -> Dict[str, str]:
    """
    This fixture provides valid parameters for testing the extract_and_validate_source_data function.

    Args:
        raw_data_schema: A fixture representing the raw data schema.

    Returns:
        A dictionary containing valid parameters for extract_and_validate_source_data function.
    """

    valid_extract_and_valid_source_params = {
    "aws_access_key": "valid_access_key",
    "aws_secret_access_key": "valid_secret_key",
    "source_bucket": "valid_source_bucket",
    "source_s3_key": "valid_source_s3_key",
    "raw_bucket": "valid_raw_bucket",
    "raw_bucket_weekly_dump_prefix": "valid_raw_weekly_dump_prefix",
    "batch_year": "2022",
    "batch_week": "48",
    "chunk_size": 100000,
    "raw_data_schema": raw_data_schema
    }

    return valid_extract_and_valid_source_params

@pytest.fixture(scope="class")
def valid_raw_s3_config(aws_credentials: dict) -> Dict[str, str]:
    """
    This fixture provides valid S3 configuration parameters.

    Returns:
        A dictionary containing valid S3 configuration parameters.
    """

    valid_raw_s3_config = {
        "access_key": aws_credentials['aws_access_key'],
        "secret_key": aws_credentials['aws_secret_key'],
        'source_bucket': aws_credentials['source_bucket'],
        'source_s3_key': aws_credentials['source_s3_key'],
        "raw_bucket": aws_credentials['raw_bucket'],
        "raw_bucket_folder": aws_credentials['raw_bucket_folder'],
        "raw_bucket_metadata_prefix": aws_credentials['raw_bucket_metadata_prefix'],
        "raw_bucket_weekly_dump_prefix": aws_credentials['raw_bucket_weekly_dump_prefix']
    }

    return valid_raw_s3_config

@pytest.fixture(scope="class")
def valid_transformed_s3_config(aws_credentials: dict) -> Dict[str, str]:

    valid_transformed_s3_config = {
            'transformed_bucket': aws_credentials['transformed_bucket'],
                'raw_bucket': aws_credentials['raw_bucket'],
                'access_key': aws_credentials['aws_access_key'],
                'secret_key': aws_credentials['aws_secret_key'],
                'raw_s3_key': aws_credentials['raw_s3_key'],
                # 'base_prefix': aws_credentials['transformed_bucket'],
                'source_bucket': aws_credentials['source_bucket'],
                'source_s3_key': aws_credentials['source_s3_key'],
                'raw_bucket_folder': aws_credentials['raw_bucket_folder'],
                'raw_bucket_metadata_prefix': aws_credentials['raw_bucket_metadata_prefix'],
                'raw_bucket_weekly_dump_prefix': aws_credentials['raw_bucket_weekly_dump_prefix'],
                'transformed_bucket_metadata_prefix': aws_credentials['transformed_bucket_metadata_prefix']
        }

    return valid_transformed_s3_config

@pytest.fixture(scope="class")
def make_parquet_bytes() -> Callable[[pd.DataFrame], bytes]:
    def _make(df: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        print("df_input", df)
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.read()
    return _make

# @pytest.fixture(scope="class")
# def upload_raw_data_to_raw_bucket(
#                                     moto_boto3_client: Session,
#                                     create_transformed_bucket: Dict,
#                                 ) -> None:
#     """
#     This function uploads a sample transformed parquet file to the mock transformed S3 bucket.
#     Args:
#         moto_boto3_client: Fixture providing a mocked boto3 client.
#         aws_credentials: Fixture providing mocked AWS credentials.
#     Returns:
#         None
#     """

#     def _upload(df: pd.DataFrame) -> None:
#         """
#         This function uploads the provided DataFrame as a parquet file to the mock transformed S3 bucket.

#         Args: 
#             df (pd.DataFrame): The DataFrame to be uploaded as a parquet file.
        
#         Returns:
#             None
#         """

#         buffer = BytesIO()
#         df.to_parquet(buffer, index=False)
#         buffer.seek(0)
#         s3_config = create_transformed_bucket
#         bucket = s3_config['raw_bucket']
#         # processed_s3_key = s3_config['transformed_s3_key']
#         moto_boto3_client.put_object(
#             Bucket=bucket,
#             Key=processed_s3_key,
#             Body=buffer.read()
#         )
#     return _upload


class TestExtractAndValidateSourceData:
    
    def test_invalid_aws_params(self, #still review this function later, make it dynamic for multiple parameters
                                extract_and_valid_source_params):
        invalid_aws_params_types = [123, 456.78, None, [], {}, ()]
        valid_batch_year = extract_and_valid_source_params["batch_year"]
        valid_batch_week = extract_and_valid_source_params["batch_week"]
        chunk_size = extract_and_valid_source_params["chunk_size"]
        valid_raw_data_schema = extract_and_valid_source_params['raw_data_schema']
        
        for invalid_param in invalid_aws_params_types:
            with pytest.raises(Exception) as exc_info:
                result = extract_and_validate_source_data(
                    aws_access_key=invalid_param,
                    aws_secret_access=invalid_param,
                    source_bucket=invalid_param,
                    source_s3_key=invalid_param,
                    raw_bucket=invalid_param,
                    raw_bucket_weekly_dump_prefix=invalid_param,
                    batch_year=valid_batch_year,
                    batch_week=valid_batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=valid_raw_data_schema
                )

            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
            assert exc_info.value.args[0]['error'] == "All AWS parameters must be of type string"

    def test_invalid_batch_year(self, extract_and_valid_source_params) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when batch_year is not a string.

        It tests various invalid types for batch_year including integer, float, None, list,tuple, and dict. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_batch_years = [100,
                                    100.5,
                                    None,
                                        [],
                                        (), {}]
        
        valid_aws_access_key_type = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key_type = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket_type = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key_type = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket_type = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        chunk_size = extract_and_valid_source_params["chunk_size"]
        batch_week = extract_and_valid_source_params["batch_week"]
        valid_raw_data_schema = extract_and_valid_source_params['raw_data_schema']

        for batch_year in invalid_batch_years:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key_type,
                    aws_secret_access=valid_aws_secret_key_type,
                    source_bucket=valid_source_bucket_type,
                    source_s3_key=valid_source_s3_key_type,
                    raw_bucket=valid_raw_bucket_type,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
            assert exc_info.value.args[0]['error'] == "batch_year must be a string"
    
    def test_invalid_batch_week(self, extract_and_valid_source_params) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when batch_week is not a string.

        It tests various invalid types for batch_week including integer, float, None, list,tuple, and dict. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_batch_weeks = [100,100.5,None,
                                [], (), {}]
        
        valid_aws_access_key_type = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key_type = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket_type = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key_type = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket_type = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        chunk_size = extract_and_valid_source_params["chunk_size"]
        batch_year = extract_and_valid_source_params["batch_year"]
        valid_raw_data_schema = extract_and_valid_source_params['raw_data_schema']

        for batch_week in invalid_batch_weeks:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key_type,
                    aws_secret_access=valid_aws_secret_key_type,
                    source_bucket=valid_source_bucket_type,
                    source_s3_key=valid_source_s3_key_type,
                    raw_bucket=valid_raw_bucket_type,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
            assert exc_info.value.args[0]['error'] == "batch_week must be a string"
    
    def test_invalid_chunk_size(self, extract_and_valid_source_params) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when chunk_size is not an integer.

        It tests various invalid types for chunk_size including string, float, None, list,tuple, and dict. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_chunk_sizes = ["100000",
                                    100000.5,
                                    None,
                                        [],
                                        (), {}]
        
        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        batch_year = extract_and_valid_source_params["batch_year"]
        batch_week = extract_and_valid_source_params["batch_week"]
        valid_raw_data_schema = extract_and_valid_source_params['raw_data_schema']

        for chunk_size in invalid_chunk_sizes:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_bucket=valid_raw_bucket,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
            assert exc_info.value.args[0]['error'] == "chunk_size must be an integer"

    def test_invalid_raw_data_schema(self, 
                                     extract_and_valid_source_params: dict) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when raw_data_schema is not a dictionary.

        It tests various invalid types for raw_data_schema including string, integer, float, None, list. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_raw_data_schemas = ["{'col1': 'str'}", 100, 100.5, 
                                    (), None, []]
        
        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_chunk_size = extract_and_valid_source_params['chunk_size']

        for raw_data_schema in invalid_raw_data_schemas:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_bucket=valid_raw_bucket,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=valid_batch_year,
                    batch_week=valid_batch_week,
                    chunk_size=valid_chunk_size,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
            assert exc_info.value.args[0]['error'] == "raw_data_schema must be a dictionary"


    @patch('src.etl.extract_source_data')
    def test_error_from_extract_source_data(self, 
                                            mock_extract_source_data: MagicMock,
                                                extract_and_valid_source_params: dict,
                                                raw_data_schema: dict
                                            ) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by extract_source_data.

        This test mocks the extract_source_data function to raise an exception and verifies that
        extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            mock_extract_source_data: A mock object for the extract_source_data function.
            raw_data_schema: A fixture providing the raw data schema.

        Returns:
            None

        """
        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_chunk_size = extract_and_valid_source_params['chunk_size']

        mock_extract_source_data.side_effect = Exception({
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": "Source extraction error occured"
            })
        
        with pytest.raises(Exception) as exc_info:
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_bucket=valid_raw_bucket,
                raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                batch_year=valid_batch_year,
                batch_week=valid_batch_week,
                chunk_size=valid_chunk_size,
                raw_data_schema=raw_data_schema
            )
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
        assert exc_info.value.args[0]['error'] == {
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": "Source extraction error occured"
            }
    
    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    def test_error_from_validate_raw_data(self,
                                        mock_extract_source_data: MagicMock,    
                                        mock_validate_raw_data: MagicMock,
                                        extract_and_valid_source_params: dict,
                                        raw_data_schema: dict
                                        ) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by validate_raw_data.

        This test provides invalid raw data that will cause validate_raw_data to raise an exception.
        It verifies that extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            mock_extract_source_data: A mock object for the extract_source_data function.
            extract_and_valid_source_params: A fixture providing valid parameters for extract_and_validate_source_data function.
            raw_data_schema: A fixture providing the raw data schema.

        Returns:
            None

        """
        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_chunk_size = extract_and_valid_source_params['chunk_size']

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {valid_source_bucket}, s3 key: {valid_source_s3_key}",
            "data": pd.DataFrame({
                "col1": ["valid_string", "another_string"],
                "col2": ["invalid_int", "also_invalid"]
            })
        }
        missing_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]
        mock_validate_raw_data.side_effect = Exception({
                "status": "error",
                "message": "An error occurred during raw data validation",
                "error": {
                    "status": "error",
                    "message": "column validation error occured",
                    "error": f"columns: {sorted(missing_cols)} are missing"
                }
        })


        with pytest.raises(Exception) as exc_info:            
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_bucket=valid_raw_bucket,
                raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                batch_year=valid_batch_year,
                batch_week=valid_batch_week,
                chunk_size=valid_chunk_size,
                raw_data_schema=raw_data_schema                
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
        
        expected_validating_raw_data_error = {
                "status": "error",
                "message": "An error occurred during raw data validation",
                "error": {
                "status": "error",
                "message": "column validation error occured",
                "error": f"columns: {sorted(missing_cols)} are missing"
            }
        }
        assert exc_info.value.args[0]['error'] == expected_validating_raw_data_error
    

    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    @patch('src.etl.boto3.client')
    def test_error_from_boto3_client(self,
                            mock_boto3_client: BaseClient,
                            mock_extract_source_data: MagicMock,
                            mock_validate_raw_data: MagicMock,
                            extract_and_valid_source_params: dict,
                            raw_data_schema: dict) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by the boto3 client.

        This test mocks the boto3 client to raise an exception when called and verifies that
        extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            extract_and_valid_source_params: A fixture providing valid parameters for extract_and_validate_source_data function.
            raw_data_schema: A fixture providing the raw data schema.

        Returns:
            None

        """

        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        batch_year = extract_and_valid_source_params['batch_year']
        batch_week = extract_and_valid_source_params['batch_week']
        chunk_size = extract_and_valid_source_params['chunk_size']

        extracted_df_sample = pd.DataFrame({
                "ride_id": ["ride123", "ride124"],
                "rideable_type": ["electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2"],
                "start_station_id": [1.0, 2.0],
                "end_station_name": [np.nan, np.nan],
                "end_station_id": [np.nan, np.nan],
                "start_lat": [40.7128, 34.0522],
                "start_lng": [-74.0060, -118.2437],
                "end_lat": [np.nan, np.nan],
                "end_lng": [np.nan, np.nan],
                "member_casual": ["member", "casual"],
            })

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {valid_source_bucket}, s3 key: {valid_source_s3_key}",
            "data": extracted_df_sample
        }

        mock_validate_raw_data.return_value = {
            "status": "success",
            "message": "Raw data validation completed successfully",
            "data_shape": extracted_df_sample.shape
        }

        mock_boto3_client.side_effect = Exception("An error occurred with the boto3 client")
        
        with pytest.raises(Exception) as exc_info:
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_bucket=valid_raw_bucket,
                raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                batch_year=batch_year,
                batch_week=batch_week,
                chunk_size=chunk_size,
                raw_data_schema=raw_data_schema
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
        assert exc_info.value.args[0]['error'] == "An error occurred with the boto3 client"

    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    @patch('src.etl.boto3.client')
    def test_error_from_boto3_put_object(self,
                            mock_boto3_client: BaseClient,
                            mock_extract_source_data: MagicMock,
                            mock_validate_raw_data: MagicMock,
                            extract_and_valid_source_params: dict,
                            raw_data_schema: dict) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by the boto3 client's put_object method.

        This test mocks the boto3 client's put_object method to raise an exception when called and verifies that
        extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            extract_and_valid_source_params: A fixture providing valid parameters for extract_and_validate_source_data function.
            raw_data_schema: A fixture providing the raw data schema.

        Returns:
            None

        """
        valid_aws_access_key = extract_and_valid_source_params['aws_access_key']
        valid_aws_secret_key = extract_and_valid_source_params['aws_secret_access_key']
        valid_source_bucket = extract_and_valid_source_params['source_bucket']
        valid_source_s3_key = extract_and_valid_source_params['source_s3_key']
        valid_raw_bucket = extract_and_valid_source_params['raw_bucket']
        valid_raw_bucket_weekly_dump_prefix = extract_and_valid_source_params['raw_bucket_weekly_dump_prefix']
        batch_year = extract_and_valid_source_params['batch_year']
        batch_week = extract_and_valid_source_params['batch_week']
        chunk_size = extract_and_valid_source_params['chunk_size']
        extracted_df_sample = pd.DataFrame({
                "ride_id": ["ride123", "ride124"],
                "rideable_type": ["electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2"],
                "start_station_id": [1.0, 2.0],
                "end_station_name": [np.nan, np.nan],
                "end_station_id": [np.nan, np.nan],
                "start_lat": [40.7128, 34.0522],
                "start_lng": [-74.0060, -118.2437],
                "end_lat": [np.nan, np.nan],
                "end_lng": [np.nan, np.nan],
                "member_casual": ["member", "casual"],
            })
        
        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {valid_source_bucket}, s3 key: {valid_source_s3_key}",
            "data": extracted_df_sample
        }

        mock_validate_raw_data.return_value = {
            "status": "success",
            "message": "Raw data validation completed successfully",
            "data_shape": extracted_df_sample.shape
        }
        mock_boto3_client.return_value.put_object.side_effect = Exception("An error occurred with while trying to dump a file to the raw bucket using boto3 client's put_object method")
        with pytest.raises(Exception) as exc_info:
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_bucket=valid_raw_bucket,
                raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                batch_year=batch_year,
                batch_week=batch_week,
                chunk_size=chunk_size,
                raw_data_schema=raw_data_schema
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating source data"
        assert exc_info.value.args[0]['error'] == "An error occurred with while trying to dump a file to the raw bucket using boto3 client's put_object method"


    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    def test_successful_extraction_and_validation(self,
                                                    mock_extract_source_data: MagicMock,
                                                    mock_validate_raw_data: MagicMock,
                                                    mock_boto3_client: BaseClient,
                                                    extract_and_valid_source_params: dict,
                                                    valid_raw_s3_config: dict,
                                                    valid_sample_raw_df: Dict,
                                                    raw_data_schema: dict,
                                                    create_raw_bucket: None,
                                                    create_source_bucket: None,) -> None:
            """
            Test that extract_and_validate_source_data successfully extracts and validates data.
            This test provides valid parameters and verifies that the function returns a success status
            along with the expected data.

            Args:
                mock_extract_source_data (MagicMock): Mocked extract_source_data function
                mock_validate_raw_data (MagicMock): Mocked validate_raw_data function
                raw_data_schema (dict): Schema for raw data validation
                mock_boto3_client (BaseClient): Mocked boto3 client
            
            Returns:
                None
            """

            extracted_df_sample = valid_sample_raw_df

            valid_aws_access_key = valid_raw_s3_config['access_key']
            valid_aws_secret_key = valid_raw_s3_config['secret_key']
            valid_source_bucket = valid_raw_s3_config['source_bucket']
            valid_source_s3_key = valid_raw_s3_config['source_s3_key']
            valid_raw_bucket = valid_raw_s3_config['raw_bucket']
            valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
            batch_year = extract_and_valid_source_params['batch_year']
            batch_week = extract_and_valid_source_params['batch_week']
            chunk_size = extract_and_valid_source_params['chunk_size']

            mock_extract_source_data.return_value = {
                "status": "success",
                "message": "Successfully extracted data from source s3 bucket",
                "data": extracted_df_sample
            }

            mock_validate_raw_data.return_value = {
                "status": "success",
                "message": "Raw data validation successful",
                "data_shape": extracted_df_sample.shape
            }

            with patch('src.etl.boto3.client', side_effect = mock_boto3_client):
                result = extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_bucket=valid_raw_bucket,
                raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                batch_year=batch_year,
                batch_week=batch_week,
                chunk_size=chunk_size,
                raw_data_schema=raw_data_schema
            )
            
            assert result['status'] == "success"
            assert result['message'] == "Successfully extracted and validated raw data"
            assert result['extracted_dump_s3_key'] == f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{batch_year}_week_{batch_week}.csv"
            assert result['batch_year'] == batch_year
            assert result['batch_week'] == batch_week


class TestLoadRawDataToS3:
    """
    Test suite for the load_raw_data_to_s3 function.
    """

    def test_invalid_raw_s3_config_arg(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when s3_config is not a dictionary.

        It tests various invalid types for s3_config parameters including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_raw_s3_config_types = [123, 100.5, None, [], (), "invalid_type"]

        for invalid_raw_s3_config in invalid_raw_s3_config_types:
            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=invalid_raw_s3_config)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "s3_config must be a dictionary"
    
    def test_invalid_access_key_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when aws_access_key is not a string.

        It tests various invalid types for aws_access_key including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_access_key_types = [123, 100.5, None, [], (), "invalid_type"]

        valid_raw_s3_config = {
            "aws_access_key": "valid_access_key",
            "aws_secret_access_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_s3_key": "valid_raw_s3_key",
            "source_bucket": "valid_source_bucket",
            "source_s3_key": "valid_source_s3_key"
        }

        for invalid_access_key in invalid_access_key_types:
            s3_config_with_invalid_access_key = valid_raw_s3_config.copy()
            s3_config_with_invalid_access_key['aws_access_key'] = invalid_access_key

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_access_key)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "access_key must be a string"
    
    def test_invalid_access_key_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when aws_access_key is not a string.

        It tests various invalid types for aws_access_key including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_access_key_types = [123, 100.5, None, [], (), "invalid_type"]

        valid_raw_s3_config = {
            "aws_access_key": "valid_access_key",
            "aws_secret_access_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_s3_key": "valid_raw_s3_key",
            "source_bucket": "valid_source_bucket",
            "source_s3_key": "valid_source_s3_key"
        }

        for invalid_access_key in invalid_access_key_types:
            s3_config_with_invalid_access_key = valid_raw_s3_config.copy()
            s3_config_with_invalid_access_key['aws_access_key'] = invalid_access_key

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_access_key)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "access_key must be a string"

    def test_invalid_secret_key_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when aws_secret_access_key is not a string.

        It tests various invalid types for aws_secret_access_key including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_secret_key_types = [123, 100.5, None, [], (), {}]

        valid_raw_s3_config = {
            "access_key": "valid_access_key",
            "secret_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_bucket_folder": "valid_raw_bucket_folder",
            "raw_bucket_metadata_prefix": "valid_raw_bucket_metadata_prefix",
            "raw_bucket_weekly_dump_prefix": "valid_raw_bucket_weekly_dump_prefix"
        }

        for invalid_secret_key in invalid_secret_key_types:
            s3_config_with_invalid_secret_key = valid_raw_s3_config.copy()
            s3_config_with_invalid_secret_key['secret_key'] = invalid_secret_key

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_secret_key)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "secret_key must be a string"

    def test_invalid_raw_bucket_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when raw_bucket is not a string.

        It tests various invalid types for raw_bucket including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_raw_bucket_types = [123, 100.5, None, [], (), {}]

        valid_raw_s3_config = {
            "access_key": "valid_access_key",
            "secret_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_bucket_folder": "valid_raw_bucket_folder",
            "raw_bucket_metadata_prefix": "valid_raw_bucket_metadata_prefix",
            "raw_bucket_weekly_dump_prefix": "valid_raw_bucket_weekly_dump_prefix"
        }

        for invalid_raw_bucket in invalid_raw_bucket_types:
            s3_config_with_invalid_raw_bucket = valid_raw_s3_config.copy()
            s3_config_with_invalid_raw_bucket['raw_bucket'] = invalid_raw_bucket

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_raw_bucket)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "raw_bucket must be a string"

    def test_invalid_raw_bucket_folder_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when raw_bucket_folder is not a string.

        It tests various invalid types for raw_bucket_folder including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_raw_bucket_folder_types = [123, 100.5, None, [], (), {}]

        valid_raw_s3_config = {
            "access_key": "valid_access_key",
            "secret_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_bucket_folder": "valid_raw_bucket_folder",
            "raw_bucket_metadata_prefix": "valid_raw_bucket_metadata_prefix",
            "raw_bucket_weekly_dump_prefix": "valid_raw_bucket_weekly_dump_prefix"
        }

        for invalid_raw_bucket_folder in invalid_raw_bucket_folder_types:
            s3_config_with_invalid_raw_bucket_folder = valid_raw_s3_config.copy()
            s3_config_with_invalid_raw_bucket_folder['raw_bucket_folder'] = invalid_raw_bucket_folder

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_raw_bucket_folder)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "raw_bucket_folder must be a string"
    
    def test_invalid_raw_bucket_metadata_prefix_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when raw_bucket_metadata_prefix is not a string.

        It tests various invalid types for raw_bucket_metadata_prefix including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_raw_bucket_metadata_prefix_types = [123, 100.5, None, [], (), {}]

        valid_raw_s3_config = {
            "access_key": "valid_access_key",
            "secret_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_bucket_folder": "valid_raw_bucket_folder",
            "raw_bucket_metadata_prefix": "valid_raw_bucket_metadata_prefix",
            "raw_bucket_weekly_dump_prefix": "valid_raw_bucket_weekly_dump_prefix"
        }

        for invalid_raw_bucket_metadata_prefix in invalid_raw_bucket_metadata_prefix_types:
            s3_config_with_invalid_raw_bucket_metadata_prefix = valid_raw_s3_config.copy()
            s3_config_with_invalid_raw_bucket_metadata_prefix['raw_bucket_metadata_prefix'] = invalid_raw_bucket_metadata_prefix

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_raw_bucket_metadata_prefix)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "raw_bucket_metadata_prefix must be a string"
    
    def test_invalid_raw_bucket_weekly_dump_prefix_type(
        self
    ) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when raw_bucket_weekly_dump_prefix is not a string.

        It tests various invalid types for raw_bucket_weekly_dump_prefix including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_raw_bucket_weekly_dump_prefix_types = [123, 100.5, None, [], (), {}]

        valid_raw_s3_config = {
            "access_key": "valid_access_key",
            "secret_key": "valid_secret_key",
            "raw_bucket": "valid_bucket",
            "raw_bucket_folder": "valid_raw_bucket_folder",
            "raw_bucket_metadata_prefix": "valid_raw_bucket_metadata_prefix",
            "raw_bucket_weekly_dump_prefix": "valid_raw_bucket_weekly_dump_prefix"
        }

        for invalid_raw_bucket_weekly_dump_prefix in invalid_raw_bucket_weekly_dump_prefix_types:
            s3_config_with_invalid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config.copy()
            s3_config_with_invalid_raw_bucket_weekly_dump_prefix['raw_bucket_weekly_dump_prefix'] = invalid_raw_bucket_weekly_dump_prefix

            with pytest.raises(Exception) as exc_info:
                load_to_raw_s3_obj = LoadToRawS3(s3_config=s3_config_with_invalid_raw_bucket_weekly_dump_prefix)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
            assert exc_info.value.args[0]['error'] == "raw_bucket_weekly_dump_prefix must be a string"

    @patch('src.etl.boto3.client')
    def test_error_from_initialising_boto3_client(self,
                            mock_boto3_client: BaseClient,
                            valid_raw_s3_config: Dict[str, str]
                            ) -> None:
        """
        Test that load_raw_data_to_s3 handles exceptions raised when initializing the boto3 client.

        This test mocks the boto3 client to raise an exception when called and verifies that
        load_raw_data_to_s3 catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """
        mock_boto3_client.side_effect = Exception("An error occurred while connecting to s3")

        with pytest.raises(Exception) as exc_info:
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToRawS3"
        assert exc_info.value.args[0]['error'] == "An error occurred while connecting to s3"
    
    def test_generate_partition_path_invalid_args(self,
                                                  valid_raw_s3_config: Dict[str, str]) -> None:
        """
        Test that the _generate_partition_path method raises an exception when given invalid arguments.

        It tests various invalid types for batch_year and batch_week including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """
        invalid_batch_years = [123, 100.5, None, [], (), {}]
        invalid_batch_weeks = [1, 52.5, None, [], (), {}]

        for invalid_batch_year in invalid_batch_years:
            with pytest.raises(Exception) as exc_info:
                valid_week_type = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.generate_partition_path(year=invalid_batch_year, week=valid_week_type)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating partition path"
            assert exc_info.value.args[0]['error'] == "year must be a string"

        for invalid_batch_week in invalid_batch_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year_type = "2022"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.generate_partition_path(year=valid_batch_year_type, 
                                                           week=invalid_batch_week)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating partition path"
            assert exc_info.value.args[0]['error'] == "week must be a string"
 
    def test_generate_partition_path_success(self,
                                              valid_raw_s3_config: Dict[str, str],
                                              extract_and_valid_source_params: Dict[str, str]) -> None:
        """
        Test that the _generate_partition_path method generates the correct partition path when given valid arguments.

        The test verifies that the generated partition path matches the expected format based on the provided batch_year and batch_week.

        Args:
            None

        Returns:
            None
        """
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_raw_bucket_folder = valid_raw_s3_config['raw_bucket_folder']
        expected_partition_path = f"{valid_raw_bucket_folder}/year={valid_batch_year}/week={valid_batch_week}/ingestion_ts={datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')}/"
        generated_partition_path = load_to_raw_s3_obj.generate_partition_path(year=valid_batch_year, week=valid_batch_week)

        assert generated_partition_path == expected_partition_path

    def test_get_batch_filename_invalid_params(
        self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_batch_filename method raises an exception when given invalid parameters.

        It tests various invalid types for batch_year and batch_week including integer, float, None, list, and tuple. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """
        invalid_batch_years = [123, 100.5, None, [], (), {}]
        invalid_batch_weeks = [1, 52.5, None, [], (), {}]

        for invalid_batch_year in invalid_batch_years:
            with pytest.raises(Exception) as exc_info:
                valid_week_type = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.get_batch_filename(year=invalid_batch_year, week=valid_week_type)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating batch filename"
            assert exc_info.value.args[0]['error'] == "year must be a string"

        for invalid_batch_week in invalid_batch_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year_type = "2022"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.get_batch_filename(year=valid_batch_year_type, 
                                                           week=invalid_batch_week)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating batch filename"
            assert exc_info.value.args[0]['error'] == "week must be a string"
        
    def test_get_batch_filename_success(self,
                                        valid_raw_s3_config: Dict[str, str],
                                        extract_and_valid_source_params: Dict[str, str]) -> None:
        """
        Test that the _get_batch_filename method generates the correct batch filename when given valid parameters.
        The test verifies that the generated batch filename matches the expected format based on the provided batch_year and batch_week.
        Args:
            None
        Returns:
            None
        """
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        expected_batch_filename = f"trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
        generated_batch_filename = load_to_raw_s3_obj.get_batch_filename(year=valid_batch_year, week=valid_batch_week)


    def test_get_ride_ids_from_metadata_invalid_params(
        self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_ride_ids_from_metadata method raises an exception when given invalid parameters.
        It tests various invalid types for batch_year and batch_week including integer, float, None, list, and tuple.
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_batch_years = [123, 100.5, None, [], (), {}]
        invalid_batch_weeks = [1, 52.5, None, [], (), {}]

        for invalid_batch_year in invalid_batch_years:
            with pytest.raises(Exception) as exc_info:
                valid_week_type = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=invalid_batch_year, partition_week=valid_week_type)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {invalid_batch_year}, partition week: {valid_week_type}"
            assert exc_info.value.args[0]['error'] == "partition_year must be a string"

        for invalid_batch_week in invalid_batch_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year_type = "2022"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year_type, 
                                                              partition_week=invalid_batch_week)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year_type}, partition week: {invalid_batch_week}"
            assert exc_info.value.args[0]['error'] == "partition_week must be a string"

    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metadata_get_object_error(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_ride_ids_from_metadata method handles exceptions raised when calling boto3 client's get_object method.

        This test mocks the boto3 client's get_object method to raise an exception and verifies that
        _get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.side_effect = Exception("An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method")

        with pytest.raises(Exception) as exc_info:
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            get_ride_ids = load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year, partition_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
        assert exc_info.value.args[0]['error'] == "An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method"
    
    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metatdata_invalid_parquet_byte(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_ride_ids_from_metadata method handles exceptions raised when the boto3 client's get_object method returns a response with a non parquet byte body or file is corrupted.

        This test mocks the boto3 client's get_object method to return a response with an invalid body and verifies that
        _get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.return_value = {
            "Body": io.BytesIO(b"non_parquet_bytes")
        }

        with pytest.raises(Exception) as exc_info:
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year, partition_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
        expected_error = "Either the file is corrupted or this is not a parquet file."
        assert expected_error in exc_info.value.args[0]['error']
    
    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metadata_invalid_body(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_ride_ids_from_metadata method handles the case when the boto3 client's get_object method returns a response with an invalid body which is not a IO byte stream.

        This test mocks the boto3 client's get_object method to return a response with an empty parquet file and verifies that
        _get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """

        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.return_value = {
            "Body": "invalid_body_not_io_byte_stream_obj"
        }

        with pytest.raises(Exception) as exc_info: 
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                            partition_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
        assert exc_info.value.args[0]['error'] == "'str' object has no attribute 'read'"
    
    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metadata_network_interrupt_while_reading_body(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _get_ride_ids_from_metadata method handles exceptions raised when there is a network interruption while reading the body of the response from the boto3 client's get_object method.

        This test mocks the boto3 client's get_object method to return a response with a body that raises an exception when read and verifies that
        _get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """

        mock_body = MagicMock()
        mock_body.read.side_effect = IOError("Connection error occurred while trying to read the body of the response from boto3 client's get_object method")

        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.return_value = {
            "Body": mock_body
        }

        with pytest.raises(Exception) as exc_info: 
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                            partition_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
        assert exc_info.value.args[0]['error'] == "Connection error occurred while trying to read the body of the response from boto3 client's get_object method"

    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metadata_missing_primary_key(self,
        mock_boto3_client: BaseClient,
        make_parquet_bytes,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        
        """
        Test that the _get_ride_ids_from_metadata method handles the case when the parquet file retrieved from the boto3 client's get_object method does not contain the expected keys.

        This test mocks the boto3 client's get_object method to return a response with a parquet file that is missing the expected keys and verifies that
        _get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """

        df = pd.DataFrame({
            "unexpected_key": [1, 2, 3]
        })
        df_buffer = make_parquet_bytes(df)

        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.return_value = {
            "Body": io.BytesIO(df_buffer)
        }

        with pytest.raises(Exception) as exc_info: 
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                            partition_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occured retrieving ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
        assert exc_info.value.args[0]['error'] == "The required column 'id' is missing from the metadata parquet file"
    
    @patch('src.etl.boto3.client')
    def test_get_ride_ids_from_metadata_file_with_no_ride_ids(self,
        mock_boto3_client: BaseClient,
        make_parquet_bytes: bytes,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Tests the get_ride_ids_from_metadata method for a situation where the  

        This test mocks the boto3 client's get_object method to return a response with a parquet file that contains an empty 'id' column and verifies that
        get_ride_ids_from_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            make_parquet_bytes: A fixture that converts a pandas DataFrame to parquet bytes.
            aws_credentials: A fixture that provides mock AWS credentials.
        
        Returns:
            None
        """

        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "NoSuchKey",
                    "Message": "The specified key does not exist."
                }
            },
            operation_name="GetObject"
        )

        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        load_to_raw_s3_status = load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                            partition_week=valid_batch_week)

        assert load_to_raw_s3_status['status'] == "success"
        assert load_to_raw_s3_status['message'] == f"No ride IDs found for partition year: {valid_batch_year}, partition week: {valid_batch_week}, returning empty set"
        assert load_to_raw_s3_status['ride_ids'] == set()


    def test_get_ride_ids_from_metadata_success_empty_metadata(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        create_raw_bucket: None
    ) -> None:
        
        """
        Tests the get_ride_ids_from_metadata method for a successful retrieval of ride IDs from the metadata parquet file.

        This test mocks the boto3 client's get_object method to return a response with a parquet file that contains an 'id' column with ride IDs and verifies that
        get_ride_ids_from_metadata successfully retrieves the ride IDs and returns them in the expected format.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            make_parquet_bytes: A fixture that converts a pandas DataFrame to parquet bytes.
            aws_credentials: A fixture that provides mock AWS credentials.
        
        Returns:
            None
        """
        with patch('src.etl.boto3.client', side_effect= mock_boto3_client):
           
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_status = load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                                partition_week=valid_batch_week)
            expected_ride_ids = set()#no data has been uploaded to the metadata yet, so the expected ride ids set is empty
            assert load_to_raw_s3_status['status'] == "success"
            assert load_to_raw_s3_status['message'] == f"No ride IDs found for partition year: {valid_batch_year}, partition week: {valid_batch_week}, returning empty set"
            assert load_to_raw_s3_status['ride_ids'] == expected_ride_ids
    
    def test_update_ride_ids_metadata_invalid_params(self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _update_ride_ids_metadata method raises an exception when given invalid parameters.
        It tests various invalid types for partition_year, partition_week, and ride_ids including integer, float, None, list, and tuple.
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_partition_years = [123, 100.5, None, [], (), {}]
        invalid_partition_weeks = [1, 52.5, None, [], (), {}]
        invalid_new_ride_ids = [123, 100.5, None, "not_a_set", [], (), {}]

        for invalid_partition_year in invalid_partition_years:
            with pytest.raises(Exception) as exc_info:
                valid_partition_week_type = "1"
                valid_ride_ids_set = set()
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=invalid_partition_year, partition_week=valid_partition_week_type, new_ride_ids=valid_ride_ids_set)
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {invalid_partition_year}, partition_week: {valid_partition_week_type}"
            assert exc_info.value.args[0]['error'] == "partition_year must be a string"

        for invalid_partition_week in invalid_partition_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_partition_year_type = "2022"
                valid_ride_ids_set = set()
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year_type, 
                                                              partition_week=invalid_partition_week,
                                                              new_ride_ids=valid_ride_ids_set)
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {valid_partition_year_type}, partition_week: {invalid_partition_week}"
            assert exc_info.value.args[0]['error'] == "partition_week must be a string"
        
        for invalid_ride_id in invalid_new_ride_ids:
            with pytest.raises(Exception) as exc_info:
                valid_partition_year_type = "2022"
                valid_partition_week_type = "1"
                load_to_raw_s3_obj = LoadToRawS3(
                    s3_config=valid_raw_s3_config
                )
                load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year_type,
                                                              partition_week=valid_partition_week_type,
                                                                new_ride_ids=invalid_ride_id)
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {valid_partition_year_type}, partition_week: {valid_partition_week_type}"
            assert exc_info.value.args[0]['error'] == "new_ride_ids must be a set"

    @patch('src.etl.boto3.client')
    def test_update_ride_ids_metadata_error_from_get_ride_ids_metadata(
            self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
            
    ) -> None:
        """
        Test that the _update_ride_ids_metadata method handles exceptions raised when calling the get_ride_ids_from_metadata method.

        This test mocks the get_ride_ids_from_metadata method to raise an exception and verifies that
        update_ride_ids_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.
        
        Returns:
            None

        """

        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.side_effect = Exception("An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method")

        with pytest.raises(Exception) as exc_info:
            valid_partition_year = extract_and_valid_source_params['batch_year']
            valid_partition_week = extract_and_valid_source_params['batch_week']
            new_ride_ids_set = {1, 2, 3}
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year,
                                                        partition_week=valid_partition_week,
                                                        new_ride_ids=new_ride_ids_set)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {valid_partition_year}, partition_week: {valid_partition_week}"
        assert exc_info.value.args[0]['error'] == {
            "status": "error",
            "message": f"An error occured retrieving ride IDs for partition year: {valid_partition_year}, partition week: {valid_partition_week}",
            "error": "An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method"
        }
    
    def test_update_ride_ids_metadata_invalid_retrieved_ride_ids_type(
        self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _update_ride_ids_metadata method handles the case when the get_ride_ids_from_metadata method returns an invalid type for ride IDs.

        This test mocks the get_ride_ids_from_metadata method to return a response with an invalid type for ride IDs and verifies that
        update_ride_ids_metadata catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None

        """
        invalid_ride_ids = [123, 100.5, None, "not_a_set", [], (), {}]
        for invalid_ride_id in invalid_ride_ids:
            with patch.object(LoadToRawS3, 'get_ride_ids_from_metadata', 
            return_value = {
                "status": "success",
                "message": "Successfully retrieved ride IDs",
                "ride_ids": invalid_ride_id
            }):
                with pytest.raises(Exception) as exc_info:
                    valid_partition_year = extract_and_valid_source_params['batch_year']
                    valid_partition_week = extract_and_valid_source_params['batch_week']
                    new_ride_ids_set = {1, 2, 3}
                    load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                    load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year,
                                                                partition_week=valid_partition_week,
                                                                new_ride_ids=new_ride_ids_set)
            type_invalid_ride_id = type(invalid_ride_id).__name__
        
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {valid_partition_year}, partition_week: {valid_partition_week}"
            assert exc_info.value.args[0]['error'] == f"'{type_invalid_ride_id}' object has no attribute 'union'"
    
    @patch('src.etl.boto3.client')
    def test_update_ride_ids_metadata_error_from_put_object(
        self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        
        """
        Test that the _update_ride_ids_metadata method handles exceptions raised when calling boto3 client's put_object method to update the metadata in s3.

        This test mocks the boto3 client's put_object method to raise an exception and verifies that
        update_ride_ids_metadata catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.

        Returns:
            None

        """
        with patch.object(LoadToRawS3, 'get_ride_ids_from_metadata',
        return_value = {
            "status": "success",
            "message": "Successfully retrieved ride IDs",
            "ride_ids": set()
        }):
            mock_s3_client = mock_boto3_client.return_value
            mock_s3_client.put_object.side_effect = Exception("An error occurred while trying to update metadata in s3 using boto3 client's put_object method")

            with pytest.raises(Exception) as exc_info:
                valid_partition_year = extract_and_valid_source_params['batch_year']
                valid_partition_week = extract_and_valid_source_params['batch_week']
                new_ride_ids_set = {1, 2, 3}
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year,
                                                            partition_week=valid_partition_week,
                                                            new_ride_ids=new_ride_ids_set)

            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while updating ride IDs metadata for partition_year: {valid_partition_year}, partition_week: {valid_partition_week}"
            assert exc_info.value.args[0]['error'] == "An error occurred while trying to update metadata in s3 using boto3 client's put_object method"
    
  
    def test_update_ride_ids_metadata_success(
        self,
        mock_boto3_client: BaseClient,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Test that the _update_ride_ids_metadata method successfully updates the ride IDs metadata in s3 when given valid parameters and no exceptions are raised.

        This test mocks the get_ride_ids_from_metadata method to return an empty set of ride IDs and mocks the boto3 client's put_object method to successfully update the metadata in s3. 
        It verifies that update_ride_ids_metadata returns the expected success message.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            aws_credentials: Mock AWS credentials.  
        Returns:
            None

        """
        retrieved_ride_ids_set = set()
        valid_partition_year = extract_and_valid_source_params['batch_year']
        valid_partition_week = extract_and_valid_source_params['batch_week']
        new_ride_ids_set = set(valid_sample_raw_df['ride_id'])
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            update_metadata_status = load_to_raw_s3_obj.update_ride_ids_metadata(partition_year=valid_partition_year,
                                                                partition_week=valid_partition_week,
                                                                new_ride_ids=new_ride_ids_set)

        assert update_metadata_status['status'] == "success"
        assert update_metadata_status['message'] == f"successfully updated ride IDs metadata for partition_year: {valid_partition_year}, partition_week: {valid_partition_week}"
        updated_ids = new_ride_ids_set.union(retrieved_ride_ids_set)
        assert update_metadata_status['updated_ids_count'] == len(updated_ids)
    
    def test_get_ride_ids_from_metadata_success_after_metadata_update(self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame,
        extract_and_valid_source_params: Dict[str, str],
        create_raw_bucket: None
    ) -> None:
        
        """
        Tests the get_ride_ids_from_metadata method for a successful retrieval of ride IDs after the metadata has been updated with the valid sample dataframe ride ids.

        Test verifies that get_ride_ids_from_metadata successfully retrieves the updated ride IDs from the metadata and returns them in the expected format.

        Args:
            mock_boto3_client: A mocked boto3 client function.
            make_parquet_bytes: A fixture that converts a pandas DataFrame to parquet bytes.
            aws_credentials: A fixture that provides mock AWS credentials.
        
        Returns:
            None
        """
        existing_df = valid_sample_raw_df
        with patch('src.etl.boto3.client', side_effect= mock_boto3_client):
           
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_status = load_to_raw_s3_obj.get_ride_ids_from_metadata(partition_year=valid_batch_year,
                                                                                partition_week=valid_batch_week)
            expected_ride_ids = set(existing_df['ride_id'].tolist())
            assert load_to_raw_s3_status['status'] == "success"
            assert load_to_raw_s3_status['message'] == f"Successfully retrieved {len(expected_ride_ids)} ride IDs for partition year: {valid_batch_year}, partition week: {valid_batch_week}"
            assert load_to_raw_s3_status['ride_ids'] == expected_ride_ids
    
    
    def test_filter_duplicate_rides_invalid_params(self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _filter_duplicate_rides method raises an exception when given invalid parameters.
        It tests various invalid types for batch_year, batch_week and rides_df including integer, float, None, list, and tuple.
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None
        """

        invalid_batch_years = [123, 100.5, None, [], (), {}, pd.DataFrame()]
        invalid_batch_weeks = [1, 52.5, None, [], (), {}, pd.DataFrame()]
        invalid_rides_dfs = [123, 100.5, None, "not_a_dataframe", [], (), {}]

        for invalid_batch_year in invalid_batch_years:
            with pytest.raises(Exception) as exc_info:
                valid_week_type = "1"
                valid_rides_df = pd.DataFrame()
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.filter_duplicate_rides(rides_df=valid_rides_df, batch_year=invalid_batch_year, batch_week=valid_week_type)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while filtering duplicate rides for batch year: {invalid_batch_year}, batch week: {valid_week_type}"
            assert exc_info.value.args[0]['error'] == "batch_year must be a string"

        for invalid_batch_week in invalid_batch_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year_type = "2022"
                valid_rides_df = pd.DataFrame()
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.filter_duplicate_rides(rides_df=valid_rides_df, batch_year=valid_batch_year_type, 
                                                        batch_week=invalid_batch_week)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while filtering duplicate rides for batch year: {valid_batch_year_type}, batch week: {invalid_batch_week}"
            assert exc_info.value.args[0]['error'] == "batch_week must be a string"
        
        for invalid_rides_df in invalid_rides_dfs:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year_type = "2022"
                valid_batch_week_type = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.filter_duplicate_rides(rides_df=invalid_rides_df, batch_year=valid_batch_year_type, 
                                                        batch_week=valid_batch_week_type)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while filtering duplicate rides for batch year: {valid_batch_year_type}, batch week: {valid_batch_week_type}"
            assert exc_info.value.args[0]['error'] == "rides_df must be a pandas DataFrame"
        
    def test_filter_duplicate_rides_empty_rides_df(self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Test that the _filter_duplicate_rides method returns an empty DataFrame when given an empty input DataFrame.

        This test verifies that when an empty DataFrame is passed to filter_duplicate_rides, it returns an empty DataFrame without even attempting to filter duplicate rides and raising any exceptions.

        Args:
            None

        Returns:
            None

        """
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        input_rides_df = pd.DataFrame()
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        filtered_rides_status = load_to_raw_s3_obj.filter_duplicate_rides(rides_df=input_rides_df,
                                                                       batch_year=valid_batch_year,
                                                                         batch_week=valid_batch_week)
        expected_df = filtered_rides_status['filtered_rides']
        assert filtered_rides_status['status'] == "success"
        assert filtered_rides_status['message'] == f"Input rides_df dataframe is empty, hence deduplication will be skipped in the raw bucket"
        assert filtered_rides_status['input_shape'] == expected_df.shape
    
    def test_filter_duplicate_rides_error_from_get_ride_ids_metadata(self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Test that the _filter_duplicate_rides method handles exceptions raised when calling the get_ride_ids_from_metadata method.

        This test mocks the get_ride_ids_from_metadata method to raise an exception and verifies that filter_duplicate_rides catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None

        """
        with patch.object(LoadToRawS3, 'get_ride_ids_from_metadata', 
        side_effect = Exception({
            "status": "error",
            "message": "An error occurred while trying to retrieve ride IDs from metadata",
            "error": "An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method"
        })):
            with pytest.raises(Exception) as exc_info:
                valid_batch_year = extract_and_valid_source_params['batch_year']
                valid_batch_week = extract_and_valid_source_params['batch_week']
                valid_rides_df = valid_sample_raw_df

                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.filter_duplicate_rides(rides_df=valid_rides_df, batch_year=valid_batch_year, batch_week=valid_batch_week)
 
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while filtering duplicate rides for batch year: {valid_batch_year}, batch week: {valid_batch_week}"
        assert exc_info.value.args[0]['error'] == {
            "status": "error",
            "message": "An error occurred while trying to retrieve ride IDs from metadata",
            "error": "An error occurred while trying to retrieve metadata from s3 using boto3 client's get_object method"
        }
    
    def test_filter_duplicate_rides_success(
        self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame,
        create_raw_bucket: None
    ) -> None:
        """
        Test that the _filter_duplicate_rides method successfully filters out duplicate rides from the input DataFrame based on the ride IDs retrieved from metadata.

        This test mocks the get_ride_ids_from_metadata method to return a set of existing ride IDs and verifies that filter_duplicate_rides successfully filters out the duplicate rides from the input DataFrame and returns the expected success message.

        Args:
            None

        Returns:
            None

        """

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):

            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']
            
            input_rides_df = pd.DataFrame({
                "ride_id": ["ride124", "ride125", "ride126"],
                "rideable_type": ["electric_bike", "classic_bike", "electric_bike"],
                "started_at": ["2022-01-01 08:00:00", "2022-01-01 09:00:00", "2022-01-01 10:00:00"],
                "ended_at": ["2022-01-01 08:30:00", "2022-01-01 09:30:00", "2022-01-01 10:30:00"],
                "start_lat": [41.8781, 41.8810, 41.8825],
                "start_lng": [-87.6298, -87.6235, -87.6200],
                "end_lat": [41.8810, 41.8825, 41.8840],
                "end_lng": [-87.6235, -87.6200, -87.6170],
                "member_casual": ["member", "casual", "member"]
            })

            len_input_rides = len(input_rides_df['ride_id'].to_list())
            filtered_new_rides = set(input_rides_df['ride_id'].to_list()) - set(valid_sample_raw_df['ride_id'].to_list()) #ride124 and ride125 will be filtered out as duplicates
            len_filtered_new_rides = len(filtered_new_rides)

            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            filtered_rides_status = load_to_raw_s3_obj.filter_duplicate_rides(rides_df=input_rides_df,
                                                                        batch_year=valid_batch_year,
                                                                            batch_week=valid_batch_week)
            expected_filtered_rides = input_rides_df[input_rides_df['ride_id'].isin(filtered_new_rides)]
            
            assert filtered_rides_status['status'] == "success"
            assert filtered_rides_status['message'] == f"Filtered {len_filtered_new_rides} new rides from {len_input_rides} total rides"
            pd.testing.assert_frame_equal(filtered_rides_status['filtered_rides'].reset_index(drop = True), expected_filtered_rides.reset_index(drop = True))
    
    def test_upload_to_raw_bucket_invalid_params(
        self,
        valid_raw_s3_config: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that upload_to_raw_bucket method raises exceptions when given invalid parameters: [new_rides_df, partition_year, partition_week]

        This test verifies that when invalid parameters are passed to upload_to_raw_bucket, it raises the appropriate exceptions with the expected error messages.

        Args:
            None

        Returns:
            None

        """

        invalid_new_rides_df = [123, 100.5, None, "not_a_dataframe", [], (), {}]
        invalid_partition_years = [123, 100.5, None, [], (), {}, pd.DataFrame()]
        invalid_partition_weeks = [1, 52.5, None, [], (), {}, pd.DataFrame()]
        valid_df = valid_sample_raw_df

        for invalid_new_rides in invalid_new_rides_df:
            with pytest.raises(Exception) as exc_info:
                valid_partition_year = "2022"
                valid_partition_week = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=invalid_new_rides, partition_year=valid_partition_year, partition_week=valid_partition_week)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
            assert exc_info.value.args[0]['error'] == "rides_df must be a pandas DataFrame"
            
        for invalid_partition_year in invalid_partition_years:
            with pytest.raises(Exception) as exc_info:
                valid_partition_week = "1"
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_df, partition_year=invalid_partition_year, partition_week=valid_partition_week)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
            assert exc_info.value.args[0]['error'] == f"partition_year must be a string"
            
        for invalid_partition_week in invalid_partition_weeks:
            with pytest.raises(Exception) as exc_info:
                valid_partition_year = 'test'
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_df, partition_year=valid_partition_year, partition_week=invalid_partition_week)
            
            assert exc_info.value.args[0]['status'] == 'error'
            assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
            assert exc_info.value.args[0]['error'] == f"partition_week must be a string"
            
    def test_upload_to_raw_bucket_empty_dataframe(
        self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Tests that upload_to_raw_bucket method returns a success message without attempting to upload when given an empty DataFrame.

        This test verifies that when an empty DataFrame is passed to upload_to_raw_bucket, it returns the expected success message indicating that the upload was skipped due to empty data.

        Args:
            None

        Returns:
            None
        
        """

        valid_partition_year = extract_and_valid_source_params['batch_year']
        valid_partition_week = extract_and_valid_source_params['batch_week']
        empty_df = pd.DataFrame()
        load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
        upload_status = load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=empty_df, partition_year=valid_partition_year, partition_week=valid_partition_week)

        assert upload_status['status'] == "success"
        assert upload_status['message'] == f"No new rides to upload, the DataFrame is empty"
        assert upload_status['uploaded_rides_shape'] == empty_df.shape
    
    def test_upload_to_raw_bucket_error_from_generate_partition_path(self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that upload_to_raw_bucket method handles exceptions raised from the generate_partition_path method.

        This test mocks the generate_partition_path method to raise an exception and verifies that upload_to_raw_bucket catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None
        
        """
        generate_partition_path_exception = "An error occurred while generating the partition path for raw data upload"
        with patch.object(LoadToRawS3, 'generate_partition_path', side_effect=Exception(generate_partition_path_exception)):
            with pytest.raises(Exception) as exc_info:
                valid_partition_year = extract_and_valid_source_params['batch_year']
                valid_partition_week = extract_and_valid_source_params['batch_week']
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_sample_raw_df, partition_year=valid_partition_year, partition_week=valid_partition_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
        assert exc_info.value.args[0]['error'] == generate_partition_path_exception
    
    def test_upload_to_raw_bucket_error_from_get_batch_filename(
        self,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that upload_to_raw_bucket method handles exceptions raised from the get_batch_filename method.

        This test mocks the get_batch_filename method to raise an exception and verifies that upload_to_raw_bucket catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None
        """
        get_batch_filename_exception = "An error occurred while getting the batch filename for raw data upload"
        with patch.object(LoadToRawS3, 'get_batch_filename', side_effect=Exception(get_batch_filename_exception)):
            with pytest.raises(Exception) as exc_info:
                valid_partition_year = extract_and_valid_source_params['batch_year']
                valid_partition_week = extract_and_valid_source_params['batch_week']
                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_sample_raw_df, partition_year=valid_partition_year, partition_week=valid_partition_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
        assert exc_info.value.args[0]['error'] == get_batch_filename_exception

    @patch('src.etl.boto3.client')
    def test_upload_to_raw_bucket_error_from_boto3(
        self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that upload_to_raw_bucket method handles exceptions raised from the boto3 client's upload_fileobj method.

        This test mocks the boto3 client's upload_fileobj method to raise an exception and verifies that upload_to_raw_bucket catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None
        """
        boto3_upload_exception = "An error occurred while uploading the raw data to s3 using boto3 client's upload_fileobj method"
        mock_boto3_client.return_value.put_object.side_effect = Exception(boto3_upload_exception)
        
        with pytest.raises(Exception) as exc_info:
            valid_partition_year = extract_and_valid_source_params['batch_year']
            valid_partition_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_sample_raw_df, partition_year=valid_partition_year, partition_week=valid_partition_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while uploading to raw bucket"
        assert exc_info.value.args[0]['error'] == boto3_upload_exception
    
    def test_upload_to_raw_bucket_success(
        self,
        mock_boto3_client: BaseClient,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that upload_to_raw_bucket method successfully uploads the given DataFrame to the raw bucket in s3 when given valid parameters and no exceptions are raised.

        This test verifies that when a valid DataFrame is passed to upload_to_raw_bucket along with valid partition year and week, it successfully uploads the data to the raw bucket in s3 and returns the expected success message.

        Args:
            None

        Returns:
            None
        """
        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            valid_partition_year = extract_and_valid_source_params['batch_year']
            valid_partition_week = extract_and_valid_source_params['batch_week']
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            upload_status = load_to_raw_s3_obj.upload_to_raw_bucket(new_rides_df=valid_sample_raw_df, partition_year=valid_partition_year, partition_week=valid_partition_week)

        no_records = valid_sample_raw_df.shape[0]
        assert upload_status['status'] == "success"
        assert upload_status['message'] == f"Successfully uploaded {no_records} new rides to raw bucket"
        assert upload_status['uploaded_records'] == valid_sample_raw_df.shape[0]
    

    def test_load_raw_data_invalid_params(
        self,
        valid_raw_s3_config: Dict[str, str]
    ) -> None:
        """
        Tests that load_raw_data_with_deduplication method raises exceptions when given invalid parameters: [source_bucket, source_s3_key, raw_bucket, raw_s3_key, batch_year, batch_week]

        This test verifies that when invalid parameters are passed to load_raw_data_with_deduplication, it raises the appropriate exceptions with the expected error messages.

        Args:
            None

        Returns:
            None
        """

        invalid_extracted_dump_s3_key_types = [123, 100.5, None, [], (), {}, ()]
        invalid_batch_year_types = [123, 100.5, None, [], (), {}]
        invalid_batch_week_types = [1, 52.5, None, [], (), {}]

        for invalid_s3_key_type in invalid_extracted_dump_s3_key_types:
            with pytest.raises(Exception) as exc_info:
                valid_batch_year = "2025"
                valid_batch_week = "40"

                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=invalid_s3_key_type,
                                                                    batch_year=valid_batch_year, batch_week=valid_batch_week)

            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while loading raw data to raw bucket"
            assert exc_info.value.args[0]['error'] == f"extracted_dump_s3_key must be a string"
        
        for invalid_batch_year in invalid_batch_year_types:
            with pytest.raises(Exception) as exc_info:
                valid_extracted_dump_s3_key = "source_data/test.csv"
                valid_batch_week = "40"

                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                    batch_year=invalid_batch_year, batch_week=valid_batch_week)

            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occurred while loading raw data to raw bucket"
            assert exc_info.value.args[0]['error'] == f"batch_year must be a string"
        
        for invalid_batch_week in invalid_batch_week_types:
            with pytest.raises(Exception) as exc_info:
                valid_extracted_dump_s3_key = "source_data/test.csv"
                valid_batch_year = "2025"

                load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                    batch_year=valid_batch_year, batch_week=invalid_batch_week)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while loading raw data to raw bucket"
            assert exc_info.value.args[0]['error'] == f"batch_week must be a string"

    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    def test_successful_extraction_and_validation(self,
                                                    mock_extract_source_data: MagicMock,
                                                    mock_validate_raw_data: MagicMock,
                                                    mock_boto3_client: BaseClient,
                                                    extract_and_valid_source_params: dict,
                                                    valid_raw_s3_config: dict,
                                                    valid_sample_raw_df: Dict,
                                                    raw_data_schema: dict,
                                                    create_raw_bucket: None,
                                                    create_source_bucket: None,) -> None:
            """
            Test that extract_and_validate_source_data successfully extracts and validates data.
            This test provides valid parameters and verifies that the function returns a success status
            along with the expected data.

            This test is ran again here to ensure the system and raw bucket keeps record of the uploaded dump in the mocked raw bucket.

            Args:
                mock_extract_source_data (MagicMock): Mocked extract_source_data function
                mock_validate_raw_data (MagicMock): Mocked validate_raw_data function
                raw_data_schema (dict): Schema for raw data validation
                mock_boto3_client (BaseClient): Mocked boto3 client
            
            Returns:
                None
            """

            extracted_df_sample = valid_sample_raw_df

            valid_aws_access_key = valid_raw_s3_config['access_key']
            valid_aws_secret_key = valid_raw_s3_config['secret_key']
            valid_source_bucket = valid_raw_s3_config['source_bucket']
            valid_source_s3_key = valid_raw_s3_config['source_s3_key']
            valid_raw_bucket = valid_raw_s3_config['raw_bucket']
            valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
            batch_year = extract_and_valid_source_params['batch_year']
            batch_week = extract_and_valid_source_params['batch_week']
            chunk_size = extract_and_valid_source_params['chunk_size']

            mock_extract_source_data.return_value = {
                "status": "success",
                "message": "Successfully extracted data from source s3 bucket",
                "data": extracted_df_sample
            }

            mock_validate_raw_data.return_value = {
                "status": "success",
                "message": "Raw data validation successful",
                "data_shape": extracted_df_sample.shape
            }

            with patch('src.etl.boto3.client', side_effect = mock_boto3_client):
                result = extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_bucket=valid_raw_bucket,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            
            assert result['status'] == "success"
            assert result['message'] == "Successfully extracted and validated raw data"
            assert result['extracted_dump_s3_key'] == f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{batch_year}_week_{batch_week}.csv"
            assert result['batch_year'] == batch_year
            assert result['batch_week'] == batch_week

    @patch('src.etl.boto3.client')
    def test_load_raw_data_error_from_boto3_get_obj(
        self,
        mock_boto3_client: BaseClient,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        """
        Tests that load_raw_data_with_deduplication method handles exceptions raised from the boto3 client's get_object method when trying to read the extracted dump from the source bucket.

        This test mocks the boto3 client's get_object method to raise an exception and verifies that load_raw_data_with_deduplication catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None
        """
        boto3_get_object_exception = "An error occurred while trying to read the extracted dump from the source bucket using boto3 client's get_object method"
        mock_boto3_client.return_value.get_object.side_effect = Exception(boto3_get_object_exception)

        with pytest.raises(Exception) as exc_info:
            valid_extracted_dump_s3_key = "source_data/test.csv"
            valid_batch_year = extract_and_valid_source_params['batch_year']
            valid_batch_week = extract_and_valid_source_params['batch_week']

            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                batch_year=valid_batch_year, batch_week=valid_batch_week)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while loading raw data to raw bucket"
        assert exc_info.value.args[0]['error'] == boto3_get_object_exception


    def test_load_raw_data_error_from_filter_duplicate_rides(
        self,
        mock_boto3_client: MagicMock,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str]
    ) -> None:
        
        """
        Tests that load_raw_data_with_deduplication method handles exceptions raised from the filter_duplicate_rides method.

        This test mocks the filter_duplicate_rides method to raise an exception and verifies that load_raw_data_with_deduplication catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None
        """

        filter_duplicate_rides_exception = "An error occurred while trying to filter duplicate rides in the raw bucket"
        
        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            
            with patch.object(LoadToRawS3, 'filter_duplicate_rides',
                               side_effect=Exception(filter_duplicate_rides_exception)):
                
                with pytest.raises(Exception) as exc_info:
                    raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
                    valid_batch_year = extract_and_valid_source_params['batch_year']
                    valid_batch_week = extract_and_valid_source_params['batch_week']
                    valid_extracted_dump_s3_key = f"{raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
                    
                    load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                    x = load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                        batch_year=valid_batch_year,
                                                                            batch_week=valid_batch_week)
        
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while loading raw data to raw bucket"
        assert exc_info.value.args[0]['error'] == filter_duplicate_rides_exception

    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    def test_load_raw_data_error_from_update_ride_ids_metadata(
        self,
        mock_extract_source_data: MagicMock,
        mock_validate_raw_data: MagicMock,
        mock_boto3_client: MagicMock,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        raw_data_schema: dict
    ) -> None:

        """
        Tests that load_raw_data_with_deduplication method handles exceptions raised from the update_ride_ids_metadata method.

        This test mocks the update_ride_ids_metadata method to raise an exception and verifies that load_raw_data_with_deduplication catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None

        """
        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
        batch_year = extract_and_valid_source_params['batch_year']
        batch_week = extract_and_valid_source_params['batch_week']
        chunk_size = extract_and_valid_source_params['chunk_size']
        valid_aws_access_key = valid_raw_s3_config['access_key']
        valid_aws_secret_key = valid_raw_s3_config['secret_key']
        valid_source_bucket = valid_raw_s3_config['source_bucket']
        valid_source_s3_key = valid_raw_s3_config['source_s3_key']
        valid_raw_bucket = valid_raw_s3_config['raw_bucket']

        
        new_extracted_df = pd.DataFrame({
            "ride_id": ["ride128", "ride129", "ride130"],
            "rideable_type": ["electric_bike", "classic_bike", "electric_bike"],
            "started_at": ["2022-01-01 12:00:00", "2022-01-01 13:00:00", "2022-01-01 14:00:00"],
            "ended_at": ["2022-01-01 12:30:00", "2022-01-01 13:30:00", "2022-01-01 14:30:00"],
            "start_lat": [41.8855, 41.8870, 41.8885],
            "start_lng": [-87.6140, -87.6110, -87.6080],
            "end_lat": [41.8870, 41.8885, 41.8900],
            "end_lng": [-87.6110, -87.6080, -87.6050],
            "member_casual": ["member", "casual", "member"],
        })

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": "Successfully extracted data from source s3 bucket",
            "data": new_extracted_df
        }

        mock_validate_raw_data.return_value = {
            "status": "success",
            "message": "Raw data validation successful",
            "data_shape": new_extracted_df.shape
        }

        update_ride_ids_metadata_exception = "An error occurred while trying to update the ride IDs metadata in the raw bucket"
        
        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            with patch.object(
                LoadToRawS3,
                'update_ride_ids_metadata',
                side_effect = Exception(
                    update_ride_ids_metadata_exception

                )
            ):
                with pytest.raises(Exception) as exc_info:
                    extract_and_validate_source_data(
                            aws_access_key=valid_aws_access_key,
                            aws_secret_access=valid_aws_secret_key,
                            source_bucket=valid_source_bucket,
                            source_s3_key=valid_source_s3_key,
                            raw_bucket=valid_raw_bucket,
                            raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                            batch_year=batch_year,
                            batch_week=batch_week,
                            chunk_size=chunk_size,
                            raw_data_schema=raw_data_schema
                        )
                                    
                        # with pytest.raises(Exception) as exc_info:
                    raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
                    valid_batch_year = extract_and_valid_source_params['batch_year']
                    valid_batch_week = extract_and_valid_source_params['batch_week']
                    valid_extracted_dump_s3_key = f"{raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
                    
                    load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
                    x = load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                        batch_year=valid_batch_year,
                                                                            batch_week=valid_batch_week)
                                                                                    
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while loading raw data to raw bucket"
        assert exc_info.value.args[0]['error'] == update_ride_ids_metadata_exception




    @patch('src.etl.validate_raw_data')
    @patch('src.etl.extract_source_data')
    def test_load_raw_data_test_deduplication_new_data(
        self,
        mock_extract_source_data: BaseClient,
        mock_validate_raw_data: BaseClient,
        mock_boto3_client: BaseClient,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        raw_data_schema: Dict[str, str]
    ) -> None:
             
        """
        Tests that the LoadToRawS3.load_raw_data method filters out duplicates with new data extracted from the source bucket and loads it successfully 
        to the raw bucket.
        Here, the data in the raw bucket has already been loaded with the valid_sample_dataframe 
        data in a previous test in this class. Now, this validates that when a new dataframe that contains duplicate rides
        extracted and loaded to the raw bucket, the duplicates are filtered out.
        
        Args:
            mock_boto3_client (BaeClient): mocked boto3 client 
            create_raw_bucket (None): fixture to create raw bucket in the mocked s3
            valid_raw_s3_config (Dict[str, str]): valid s3 configuration parameters
            extract_and_valid_source_params (Dict[str, str]): valid parameters for extraction and validation of source data
        
        Returns:
            None
        """
        
        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
        batch_year = extract_and_valid_source_params['batch_year']
        batch_week = extract_and_valid_source_params['batch_week']
        chunk_size = extract_and_valid_source_params['chunk_size']
        valid_aws_access_key = valid_raw_s3_config['access_key']
        valid_aws_secret_key = valid_raw_s3_config['secret_key']
        valid_source_bucket = valid_raw_s3_config['source_bucket']
        valid_source_s3_key = valid_raw_s3_config['source_s3_key']
        valid_raw_bucket = valid_raw_s3_config['raw_bucket']

        extracted_df = pd.DataFrame({
            "ride_id": ["ride125", "ride126", "ride127"],
            "rideable_type": ["electric_bike", "classic_bike", "electric_bike"],
            "started_at": ["2022-01-01 09:00:00", "2022-01-01 10:00:00", "2022-01-01 11:00:00"],
            "ended_at": ["2022-01-01 09:30:00", "2022-01-01 10:30:00", "2022-01-01 11:30:00"],
            "start_lat": [41.8810, 41.8825, 41.8840],
            "start_lng": [-87.6235, -87.6200, -87.6170],
            "end_lat": [41.8825, 41.8840, 41.8855],
            "end_lng": [-87.6200, -87.6170, -87.6140],
            "member_casual": ["casual", "member", "casual"],
        })

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": "Successfully extracted data from source s3 bucket",
            "data": extracted_df
        }

        mock_validate_raw_data.return_value = {
            "status": "success",
            "message": "Raw data validation successful",
            "data_shape": extracted_df.shape
        }


        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_bucket=valid_raw_bucket,
                    raw_bucket_weekly_dump_prefix=valid_raw_bucket_weekly_dump_prefix,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            
            
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            
            load_result = load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                batch_year=valid_batch_year, batch_week=valid_batch_week)
            
            assert load_result['status'] == "success"
            assert load_result['message'] == f"Successfully loaded raw data to raw bucket for batch year: {valid_batch_year}, week: {valid_batch_week}"
            #One record: "ride125" will be a duplicate, it occurs initially in the valid_sample_df that has been uploaded to the raw bucket, uploaded_records = 2
            assert load_result['uploaded_records'] == 2 
            assert load_result['updated_ride_ids_count'] == 5 #the three initially record that is in the raw bucket metadata, plus the two records that are not duplicates
        
    
    @patch('src.etl.validate_raw_data')
    def test_load_raw_data_test_deduplication_all_duplicates(
        self,
        mock_validate_raw_data: MagicMock,
        mock_boto3_client: BaseClient,
        create_raw_bucket: None,
        valid_raw_s3_config: Dict[str, str],
        extract_and_valid_source_params: Dict[str, str],
        raw_data_schema: Dict[str, str],
        valid_sample_raw_df: pd.DataFrame
    ) -> None:
        """
        Tests that the LoadToRawS3.load_raw_data method filters out duplicates when all data extracted from the source bucket are duplicates of the data already in the raw bucket and returns a success message with 0 uploaded records.

        Tests that when a new dataframe that contains all duplicate rides extracted and loaded to the raw bucket, the duplicates are filtered out and the function returns a success message with 0 uploaded records.


        Args:
            mock_boto3_client (BaseClient): mocked boto3 client 
            create_raw_bucket (None): fixture to create raw bucket in the mocked s3
            valid_raw_s3_config (Dict[str, str]): valid s3 configuration parameters
            extract_and_valid_source_params (Dict[str, str]): valid parameters for extraction and validation of source data
            raw_data_schema (Dict[str, str]): schema for validating the raw data

        Returns:
            None
        """

        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"
        
        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            
            load_to_raw_s3_obj = LoadToRawS3(s3_config=valid_raw_s3_config)
            
            load_result = load_to_raw_s3_obj.load_raw_data(extracted_dump_s3_key=valid_extracted_dump_s3_key,
                                                                batch_year=valid_batch_year, batch_week=valid_batch_week)
                                       
            assert load_result['status'] == "success"
            assert load_result['message'] == f"No new rides to upload for year: {valid_batch_year}, week: {valid_batch_week}. All rides are duplicates or empty."
            #One record: "ride125" will be a duplicate, it occurs initially in the valid_sample_df that has been uploaded to the raw bucket, uploaded_records = 2
            assert load_result['uploaded_records'] == 0


class TestLoadToTransformedS3:
    """
    This is a test suite for the LoadToTransformedS3 class methods.

    """

    def test_invalid_raw_s3_config_arg(
        self,
    ) -> None:
        """
        Test that LoadToTransformedS3 initialize method raises an exception when s3_config argument is invalid.
        The s3_config argument must be a dictionary.
        The s3_config argument is tested for various invalid types including string, integer, float, None, list.
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            None
        
        Returns:
            None
        """
        invalid_raw_s3_configs = [
            "invalid_raw_s3_config",
            123,
            45.67,
            (),
            None,
            []
        ]
     
        for s3_config in invalid_raw_s3_configs:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config)
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == "s3_config must be a dictionary"
    
    def test_invalid_raw_s3_config_keys(
        self,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3 initialize method raises an exception when s3_config dictionary is having invalid keys.
        The s3_config dictionary must contain [access_key, secret_access_key, transformed_bucket, raw_bucket, raw_s3_key, base_prefix] keys.
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            None
        
        Returns:
            None
        """
        invalid_raw_s3_config_key_types = (123, 123.51, {}, [], None, ())

        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'access_key'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key
            
            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
  
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"

        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()

            testing_key = 'secret_key'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)

            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'source_bucket'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"

        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'source_s3_key'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'raw_bucket_folder'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        

        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'raw_bucket_metadata_prefix'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'raw_bucket_weekly_dump_prefix'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'transformed_bucket'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'raw_bucket'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'raw_s3_key'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
        
        for invalid_raw_s3_config_key in invalid_raw_s3_config_key_types:
            s3_config_copy = valid_transformed_s3_config.copy()
            testing_key = 'transformed_bucket_metadata_prefix'
            s3_config_copy[f'{testing_key}'] = invalid_raw_s3_config_key

            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config_copy)
            
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == f"{testing_key} must be a string"
        
    @patch('src.etl.boto3.client')
    def test_successful_initialization(
        self,
        mock_boto3_client: BaseClient,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3 initialize method successfully initializes with a valid s3_config dictionary.

        This test provides a valid s3_config dictionary and verifies that the LoadToTransformedS3 object is initialized with the correct attributes.

        Args:
            valid_raw_s3_config (Dict): A dictionary containing valid s3 configuration parameters.

        Returns:
            None
        """
        
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        assert loadtos3obj.s3_config == valid_transformed_s3_config
        assert loadtos3obj.s3_client == mock_boto3_client.return_value
        mock_boto3_client.assert_called_once_with(
        's3',
        aws_access_key_id=valid_transformed_s3_config['access_key'],
        aws_secret_access_key=valid_transformed_s3_config['secret_key']
    )
 
    def test_generate_partition_path_invalid_args(
        self,
        valid_transformed_s3_config: Dict,
        aws_credentials: Dict,
    ) -> None:
        """
        Test that LoadToTransformedS3.generate_partition_path method raises an exception when input arguments are invalid.
        The year, month, day, hour arguments must be strings.
        
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        invalid_args = [
            (123, "01", "15"), #non-string user_type
            ("casual", [], "04"), #non-string year
            ("casual", "2020", None), #non-string week
        ]

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for user_type, year, week in invalid_args:
            with pytest.raises(Exception) as exc_info:
                partition_path = loadtos3obj.generate_partition_path(
                    user_type=user_type,
                    year=year,
                    week=week
                )

            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating partition path"
            if not isinstance(user_type, str):
                assert exc_info.value.args[0]['error'] == "user_type must be a string"
            elif not isinstance(year, str):
                assert exc_info.value.args[0]['error'] == "year must be a string"
            elif not isinstance(week, str):
                assert exc_info.value.args[0]['error'] == "week must be a string"
            
  
    def test_generate_partition_path_successful(
        self,
        valid_transformed_s3_config: Dict,
        aws_credentials: Dict,
    ) -> None:
        """
        Test that LoadToTransformedS3.generate_partition_path method generates the correct partition path.

        This test provides valid input arguments and verifies that the generated partition path matches the expected format.

        Args:
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        test_cases = [
            ("casual", "2023", "15", "cleaned_bikeshare/user_type=casual/year=2023/week=15/"),
            ("member", "2022", "40", "cleaned_bikeshare/user_type=member/year=2022/week=40/"),
            ("Casual", "2021", "05", "cleaned_bikeshare/user_type=casual/year=2021/week=05/"), #test case-insensitivity
        ]

        for user_type, year, week, expected_path in test_cases:
            partition_path = loadtos3obj.generate_partition_path(
                user_type=user_type,
                year=year,
                week=week
            )
            assert partition_path == expected_path
    
    def test_batch_filename(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.batch_filename method generates the correct batch filename.

        This test provides a valid batch index and verifies that the generated filename matches the expected format.

        Args:
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        test_cases = [
            "01",
            "05",
            "10",
        ]

        for batch_index in test_cases:
            filename = loadtos3obj.get_batch_filename(batch_id=batch_index)

            check_match = re.fullmatch(
                rf"{batch_index}_\d{{8}}_\d{{6}}\.parquet",
                filename
            )
            assert isinstance(check_match, Match) == True

    def test_get_processed_ride_ids_invalid_args(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.get_processed_ride_ids method raises an exception when input arguments are invalid.
        The only argument in this method is a partition_type which must be a string.
        
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        invalid_partition_year_types = [
            123, 
            [], 
            None,
            (1,2), 
            {'key': 'value'},
            45.67
        ]
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for partition_year in invalid_partition_year_types:
            with pytest.raises(Exception) as exc_info:
                processed_ride_ids = loadtos3obj.get_processed_ride_ids_from_metadata(
                    partition_year=partition_year
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured retrieving processed ride IDs for partition year: {partition_year}"
            assert exc_info.value.args[0]['error'] == "partition_year must be a string"
    
    def test_get_processed_ride_ids_no_existing_record(
        self,
        mock_boto3_client: BaseClient,
        create_transformed_bucket: None,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.get_processed_ride_ids method returns an empty set when no existing records are found in the transformed bucket.
        This case exists when no data has been loaded yet to the transformed bucket.'

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mock AWS credentials.
        Returns:
            None
        """
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        valid_partition_year = "2023"

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_ride_ids = loadtos3obj.get_processed_ride_ids_from_metadata(
                partition_year=valid_partition_year
            )
            assert process_ride_ids['status'] == "success"
            assert process_ride_ids['message'] == f"No processed ride IDs found for partition year: {valid_partition_year}, returning empty set"
            assert process_ride_ids['processed_ids'] == set()
            assert len(process_ride_ids['processed_ids']) == 0
    

    @patch('src.etl.boto3.client')
    def test_get_processed_ride_ids_exception_handling(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.get_processed_ride_ids method handles exceptions raised during the retrieval process.
        This test simulates an exception during the S3 operations and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
        mock_boto3_client.get_object.side_effect = Exception("An error occured with boto3 client")
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        valid_partition_year = "2023"

        with pytest.raises(Exception) as exc_info:
            processed_ride_ids = loadtos3obj.get_processed_ride_ids_from_metadata(
                partition_year=valid_partition_year
            )
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured retrieving processed ride IDs for partition year: {valid_partition_year}"
            assert exc_info.value.args[0]['error'] == "An error occured with boto3 client"

    def test_update_processed_ride_ids_invalid_args(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.update_processed_ride_ids method raises an exception when input arguments are invalid.
        The partition_year argument must be a string and new_ride_ids argument must be a set.
        
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        invalid_args = [
            (123, {"ride1", "ride2"}), #non-string partition_year
            ("2023", ["ride1", "ride2"]), #non-set new_ride_ids
            ("2023", None), #non-set new_ride_ids
        ]
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for partition_year, new_ride_ids in invalid_args:
            with pytest.raises(Exception) as exc_info:
                update_result = loadtos3obj.update_processed_ride_ids_metadata(
                    partition_year=partition_year,
                    new_ride_ids=new_ride_ids
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured while updating processed ride IDs for partition year: {partition_year}"
            if not isinstance(partition_year, str):
                assert exc_info.value.args[0]['error'] == "partition_year must be a string"
            elif not isinstance(new_ride_ids, set):
                assert exc_info.value.args[0]['error'] == "new_ride_ids must be a set"

    @patch('src.etl.boto3.client')
    def test_update_processed_ride_ids_exception_handling(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.update_processed_ride_ids method handles exceptions raised during the update process.
        This test simulates an exception during the S3 operations and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
        mock_boto3_client.put_object.side_effect = Exception("An error occured with boto3 client")
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        valid_partition_year = "2023"
        valid_new_ride_ids = {"ride10", "ride11"}

        with pytest.raises(Exception) as exc_info:
            update_result = loadtos3obj.update_processed_ride_ids_metadata(
                partition_year=valid_partition_year,
                new_ride_ids=valid_new_ride_ids
            )
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == f"An error occured while updating processed ride IDs for partition year: {valid_partition_year}"
            assert exc_info.value.args[0]['error'] == "An error occured with boto3 client"
    
    def test_update_processed_ride_ids_successful(
        self,
        mock_boto3_client: BaseClient,
        create_transformed_bucket: None,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.update_processed_ride_ids method successfully updates processed ride IDs metadata in the transformed bucket.
        This test verifies that the method completes without raising exceptions when provided with valid inputs.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        valid_raw_s3_config = valid_transformed_s3_config
        loadtos3obj = LoadToTransformedS3(s3_config=valid_raw_s3_config)
        valid_partition_year = "2023"
        valid_new_ride_ids = {"ride10", "ride11", "ride12"}

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            update_result = loadtos3obj.update_processed_ride_ids_metadata(
                partition_year=valid_partition_year,
                new_ride_ids=valid_new_ride_ids
            )

            assert update_result['status'] == "success"
            assert update_result['message'] == f"Successfully updated processed ride IDs metadata for partition year: {valid_partition_year}"
            assert update_result['updated_ids_count'] == len(valid_new_ride_ids)
    
    def test_get_processed_ride_ids_successful(
        self,
        mock_boto3_client: BaseClient,
        create_transformed_bucket: None,
        valid_transformed_s3_config: Dict,
        aws_credentials: Dict
    ) -> None:
        
        """
        Test that LoadToTransformedS3.get_processed_ride_ids method successfully retrieves processed ride IDs from metadata files in the transformed bucket.
        This test uploads sample metadata files to the transformed bucket and verifies that the method correctly extracts the processed ride IDs.
        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
        
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        valid_partition_year = "2023"

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_ride_ids = loadtos3obj.get_processed_ride_ids_from_metadata(
                partition_year=valid_partition_year
            )
            expected_processed_ids = {"ride10", "ride11", "ride12"}
            assert process_ride_ids['status'] == "success"
            assert process_ride_ids['message'] == f"Successfully retrieved {len(process_ride_ids['processed_ids'])} processed ride IDs for partition year: {valid_partition_year}"
            assert process_ride_ids['processed_ids'] == expected_processed_ids
            assert len(process_ride_ids['processed_ids']) == len(expected_processed_ids)
    
    def test_filter_duplicate_rides_invalid_args(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.filter_duplicate_rides method raises an exception when input arguments are invalid.
        The rides_df argument must be a pandas DataFrame.
        
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        invalid_args = [
            "invalid_dataframe",
            123,
            45.67,
            None,
            [],
            {}
        ]

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for rides_df in invalid_args:
            with pytest.raises(Exception) as exc_info:
                filtered_df_result = loadtos3obj.filter_duplicate_rides(
                    rides_df=rides_df
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while filtering duplicate rides data"
            assert exc_info.value.args[0]['error'] == "rides_df must be a pandas DataFrame"
    
    def test_filter_duplicate_rides_successful(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.filter_duplicate_rides method successfully filters out duplicate rides from the input DataFrame.
        This test uploads sample metadata files to the transformed bucket to simulate existing processed ride IDs and verifies that the method correctly filters out duplicates.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        sample_data = {
            "ride_id": ["ride10", "ride11", "ride12", "ride13", "ride14"],
            "rideable_type": ["electric_bike", "docked_bike", "classic_bike", "electric_bike", "docked_bike"],
            "started_at": ["2023-04-01 08:00:00", "2023-04-01 09:00:00", "2023-04-01 10:00:00", "2023-04-01 11:00:00", "2023-04-01 12:00:00"],
            "ended_at": ["2023-04-01 08:30:00", "2023-04-01 09:30:00", "2023-04-01 10:30:00", "2023-04-01 11:30:00", "2023-04-01 12:30:00"],
            "start_lat": [41.8781, 41.8810, 41.8820, 41.8830, 41.8840],
            "start_lng": [-87.6298, -87.6300, -87.6310, -87.6320, -87.6330],
            "end_lat": [41.8850, 41.8860, 41.8870, 41.8880, 41.8890],
            "end_lng": [-87.6340, -87.6350, -87.6360, -87.6370, -87.6380],
            "member_casual": ["casual", "member", "casual", "member", "casual"]
        }
        rides_df = pd.DataFrame(sample_data)
        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            filtered_df_result = loadtos3obj.filter_duplicate_rides(
                rides_df=rides_df
            )
            filtered_rides = filtered_df_result['filtered_rides']
            filtered_ride_ids = set(filtered_rides['ride_id'].tolist())
            expected_filtered_ride_ids = {"ride13", "ride14"} #ride10, ride11, ride12 are duplicates based on existing metadata
            assert filtered_df_result['status'] == "success"
            assert filtered_df_result['message'] == f"Filtered {len(filtered_rides)} new rides from {len(rides_df)} total rides"
            assert filtered_ride_ids == expected_filtered_ride_ids
            assert len(filtered_rides) == len(expected_filtered_ride_ids)
    
    @patch('src.etl.boto3.client')
    def test_fllter_duplicate_rides_exception_handling(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.filter_duplicate_rides method handles exceptions raised during the filtering process.
        This test simulates an exception during the S3 operations and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.

        Returns:
            None
        """
        mock_boto3_client.get_object.side_effect = Exception("An error occured with boto3 client")
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        sample_data = {
            "ride_id": ["ride10", "ride11", "ride12", "ride13", "ride14"],
            "rideable_type": ["electric_bike", "docked_bike", "classic_bike", "electric_bike", "docked_bike"],
            "started_at": ["2023-04-01 08:00:00", "2023-04-01 09:00:00", "2023-04-01 10:00:00", "2023-04-01 11:00:00", "2023-04-01 12:00:00"],
            "ended_at": ["2023-04-01 08:30:00", "2023-04-01 09:30:00", "2023-04-01 10:30:00", "2023-04-01 11:30:00", "2023-04-01 12:30:00"],
            "start_lat": [41.8781, 41.8810, 41.8820, 41.8830, 41.8840],
            "start_lng": [-87.6298, -87.6300, -87.6310, -87.6320, -87.6330],
            "end_lat": [41.8850, 41.8860, 41.8870, 41.8880, 41.8890],
            "end_lng": [-87.6340, -87.6350, -87.6360, -87.6370, -87.6380],
            "member_casual": ["casual", "member", "casual", "member", "casual"]
        }
        rides_df = pd.DataFrame(sample_data)
        with pytest.raises(Exception) as exc_info:
            filtered_df_result = loadtos3obj.filter_duplicate_rides(
                rides_df=rides_df
            )
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while filtering duplicate rides data"
            assert exc_info.value.args[0]['error'] == "An error occured with boto3 client"
    
    def test_upload_partitioned_data_invalid_args(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.upload_partitioned_data method raises an exception when input arguments are invalid.
        The rides_df argument must be a pandas DataFrame.
        The partition_columns argument must be a string.
        The test verifies that the appropriate exception is raised with the expected error message.
        
        Args:
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """
        invalid_args = [
            ({}, #new_rides_df is not a dataframe, 
             "partition_column1",
             "partition_column2"
             ),
                
            (pd.DataFrame({
                "ride_id": ["ride1", "ride2"],
                "rideable_type": ["electric_bike", "docked_bike"],
                "started_at": ["2023-04-01 08:00:00", "2023-04-01 09:00:00"],
                "ended_at": ["2023-04-01 08:30:00", "2023-04-01 09:30:00"],
                "start_lat": [41.8781, 41.8810],
                "start_lng": [-87.6298, -87.6300],
                "end_lat": [41.8850, 41.8860],
                "end_lng": [-87.6340, -87.6350],
                "member_casual": ["casual", "member"]
            }),
            {},     #partition_column1 is a non-string,
            "partition_column2"     
            )
            ,

            (
                pd.DataFrame({
                "ride_id": ["ride1", "ride2"],
                "rideable_type": ["electric_bike", "docked_bike"],
                "started_at": ["2023-04-01 08:00:00", "2023-04-01 09:00:00"],
                "ended_at": ["2023-04-01 08:30:00", "2023-04-01 09:30:00"],
                "start_lat": [41.8781, 41.8810],
                "start_lng": [-87.6298, -87.6300],
                "end_lat": [41.8850, 41.8860],
                "end_lng": [-87.6340, -87.6350],
                "member_casual": ["casual", "member"]
                }
                ),
                "member_casual",
                None #partition_column2 is a non-string
            )
        ]

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for new_rides_df, partition_column1, partition_column2 in invalid_args:
            with pytest.raises(Exception) as exc_info:
                upload_result = loadtos3obj.upload_partitioned_data(
                    new_rides_df=new_rides_df,
                    partition_column1=partition_column1,
                    partition_column2=partition_column2
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "Failed to upload partitioned data to S3"
            if not isinstance(new_rides_df, pd.DataFrame):
                assert exc_info.value.args[0]['error'] == "new_rides_df must be a pandas DataFrame"
            elif not isinstance(partition_column1, str):
                assert exc_info.value.args[0]['error'] == "partition_column1 must be a string"
            elif not isinstance(partition_column2, str):
                assert exc_info.value.args[0]['error'] == "partition_column2 must be a string"
    
    @patch('src.etl.boto3.client')
    def test_upload_exception_handling(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.upload_partitioned_data method handles exceptions raised during the upload process.
        This test simulates an exception during the S3 operations and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
        mock_boto3_client.put_object.side_effect = Exception("An error occured with boto3 client")
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        sample_data = {
            "ride_id": ["ride15", "ride16"],
            "rideable_type": ["electric_bike", "docked_bike"],
            "started_at": ["2023-04-02 08:00:00", "2023-04-02 09:00:00"],
            "ended_at": ["2023-04-02 08:30:00", "2023-04-02 09:30:00"],
            "start_lat": [41.8781, 41.8810],
            "start_lng": [-87.6298, -87.6300],
            "end_lat": [41.8850, 41.8860],
            "end_lng": [-87.6340, -87.6350],
            "member_casual": ["casual", "member"]
        }
        new_rides_df = pd.DataFrame(sample_data)
        with pytest.raises(Exception) as exc_info:
            upload_result = loadtos3obj.upload_partitioned_data(
                new_rides_df=new_rides_df
            )
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "Failed to upload partitioned data to S3"
            assert exc_info.value.args[0]['error'] == "An error occured with boto3 client"
    
    def test_upload_partitioned_data_successful(
        self,
        mock_boto3_client: BaseClient,
        create_transformed_bucket: None,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.upload_partitioned_data method successfully uploads partitioned data to the transformed S3 bucket.
        This test verifies that the method completes without raising exceptions when provided with valid inputs.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
            valid_transformed_s3_config: Valid transformed S3 configuration dictionary.
        
        Returns:
            None
        """

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        sample_data = {
            "ride_id": ["ride11", "ride12", "ride13"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "started_at": ["2023-04-02 08:00:00", "2023-04-02 09:00:00", "2023-04-02 10:00:00"],
            "ended_at": ["2023-04-02 08:30:00", "2023-04-02 09:30:00", "2023-04-02 10:30:00"],
            "start_station_name": ["start_station1", "start_station2", "start_station3"],
            "start_station_id": ["start_station_id1", "start_station_id2", "start_station_id3"],
            "end_station_name": ["end_station1", "end_station2", "end_station3"],
            "end_station_id": ["end_station_id1", "end_station_id2", "end_station_id3"],
            "start_lat": ["41.8781", "41.8810", "41.8820"],
            "start_lng": ["-87.6298", "-87.6300", "-87.6310"],
            "end_lat": ["41.8850", "41.8860", "41.8870"],
            "end_lng": ["-87.6340", "-87.6350", "-87.6360"],
            "member_casual": ["casual", "member", "casual"]
        }
        new_rides_df = pd.DataFrame(sample_data)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            upload_result = loadtos3obj.upload_partitioned_data(
                new_rides_df=new_rides_df
            )
            # expected_uploaded_ride_ids = {"ride15", "ride16"} #ride12 is duplicate (has been uploaded already based on existing metadata)
            assert upload_result['status'] == "success"
            assert upload_result['uploaded_records'] == len(new_rides_df)
  
    def test_process_raw_data_invalid_raw_data_schema(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict,
        extract_and_valid_source_params: Dict,
        valid_raw_s3_config: Dict,

    ) -> None:
        """
        Tests the the LoadToTransformedS3.process_raw_data method raises an exception when an invalid raw_data_schema is provided.
        The raw_data_schema argument must be a dictionary.
        This test verifies that the appropriate exception is raised with the expected error message.

        Args:
            aws_credentials: Mocked AWS credentials.
            valid_transformed_s3_config: Fixture providing the valid transformed S3 configuration.
 
        Returns:
            None
        """
        invalid_raw_data_schemas = [
            "invalid_schema", 
            123, 
            45.67, 
            None,
            [],
            set()
        ] #raw_data_schema must be a dictionary
        
        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for raw_data_schema in invalid_raw_data_schemas:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj.process_raw_data(
                    extracted_dump_s3_key=valid_extracted_dump_s3_key,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while processing raw data"
            assert exc_info.value.args[0]['error'] == "raw_data_schema must be a dictionary"
    

    @patch('src.etl.impute_missing_station_ids')
    @patch('src.etl.clean_raw_data')
    @patch('src.etl.extract_source_data')
    def test_process_raw_data_successful(
        self,
        mock_extract_source_data: MagicMock,
        mock_clean_raw_data: MagicMock,
        mock_impute_missing_station_ids: MagicMock,
        mock_boto3_client: BaseClient,
        create_source_bucket: None,
        upload_to_source_bucket: None,
        create_raw_bucket: None,
        valid_transformed_s3_config: Dict,
        valid_raw_s3_config: Dict,
        extract_and_valid_source_params: Dict,
        raw_data_schema: Dict,
        valid_sample_raw_df: Dict
    ) -> None:
        
        """
        Test that LoadToTransformedS3.process_raw_data method successfully processes raw data from the raw S3 bucket.
        This test verifies that the method completes without raising exceptions when provided with valid inputs.

        Args:
            mock_extract_source_data: A mocked extract_source_data function.
            mock_clean_raw_data: A mocked clean_raw_data function.
            mock_impute_missing_station_ids: A mocked impute_missing_station_ids function.
            create_source_bucket: Fixture to create source bucket.
            upload_to_source_bucket: Fixture to upload sample data to source bucket.
            create_raw_bucket: Fixture to create raw bucket.
            aws_credentials: Mocked AWS credentials.
            raw_data_schema: Fixture providing the expected schema for the raw data.
        
        Returns:
            None
        """
        
        cleaned_df = pd.DataFrame({
                "ride_id": ["ride123", "ride124", "ride125"],
                "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
                "start_station_id": ['100.0', '101.0', '102.0'],
                "end_station_name": ["end_station_1", np.nan, np.nan],
                "end_station_id": ['200.0', np.nan, np.nan],
                "start_lat": [40.7128, 34.0522, 41.8781],
                "start_lng": [-74.0060, -118.2437, -87.6298],
                "end_lat": [40.7589, np.nan, np.nan],
                "end_lng": [-73.9851, np.nan, np.nan],
                "member_casual": ["member", "member", "casual"],
            })

        imputed_df = pd.DataFrame(
            {
                "ride_id": ["ride123", "ride124", "ride125"],
                "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
                "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
                "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
                "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
                "start_station_id": ["start_station_id1", "start_station_id2", "start_station_id3"],
                "end_station_name": ["end_station_1", np.nan, np.nan],
                "end_station_id": ["end_station_id1", np.nan, np.nan],
                "start_lat": ["41.8781", "41.8810", "41.8820"],
                "start_lng": ["-87.6298", "-87.6300", "-87.6310"],
                "end_lat": ["41.8850", np.nan, np.nan],
                "end_lng": ["-87.6340", np.nan, np.nan],
                "member_casual": ["member", "member", "casual"]
            }
        )

        

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket",
            "data": valid_sample_raw_df
        }

        mock_clean_raw_data.return_value = {
            "status": "success",
            "message": "Raw data cleaned successfully",
            "data": cleaned_df
        }

        mock_impute_missing_station_ids.return_value = {
            "status": "success",
            "message": "Missing station IDs imputed successfully",
            "data": imputed_df
        }

        date_time_now = datetime(2025, 12, 30, 12, 0, 0)
        expected_s3_key = f"processed_data/processed_bikeshare_{date_time_now.strftime('%Y%m%d_%H%M%S')}.parquet"
        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"


        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            extract_and_validate_source_data(
                aws_access_key=valid_raw_s3_config['access_key'],
                aws_secret_access=valid_raw_s3_config['secret_key'],
                source_bucket=valid_raw_s3_config['source_bucket'],
                source_s3_key=valid_raw_s3_config['source_s3_key'], 
                raw_bucket=valid_raw_s3_config['raw_bucket'],
                raw_bucket_weekly_dump_prefix=valid_raw_s3_config['raw_bucket_weekly_dump_prefix'],
                batch_year=extract_and_valid_source_params['batch_year'],
                batch_week=extract_and_valid_source_params['batch_week'],
                chunk_size=extract_and_valid_source_params['chunk_size'],
                raw_data_schema=raw_data_schema
            )

            loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
    
            process_result = loadtos3obj.process_raw_data(
                extracted_dump_s3_key=valid_extracted_dump_s3_key,
                raw_data_schema=raw_data_schema,
                date_time_now=date_time_now
            )

            assert process_result['status'] == "success"
            assert process_result['message'] == f"Successfully processed raw data"
            assert process_result['processed_data_key'] == expected_s3_key


    def test_process_rides_with_partitioning_invalid_args(
        self,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Tests the LoadToTransformedS3.process_rides_with_partitioning method raises an exception when input arguments are invalid.
        The argument: processed_s3_key must be a string. The test verifies that the appropriate exception is raised with the expected error message
        when invalid arguments are provided.
        
        Args:
            aws_credentials: Mocked AWS credentials.

        Returns:
            None
        """
        invalid_args = [
            {},
            123,
            45.67,
            None,
            [],
            (),
            set()
        ] #processed_data_key must be a string

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)
        for invalid_processed_data_key in invalid_args:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj.process_rides_with_partitioning(
                    processed_data_key=invalid_processed_data_key
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while trying to deduplicate and load processed data to the transformed bucket"
            assert exc_info.value.args[0]['error'] == "processed data key must be a string type"
    

    @patch('src.etl.boto3.client')
    def test_process_rides_with_partition_exception(
        self,
        mock_boto3_client: BaseClient,
        aws_credentials: Dict,
        valid_transformed_s3_config: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.process_rides_with_partitioning method handles exceptions raised during the processing and partitioning of rides data.
        This test simulates an exception during the S3 operations and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """

        mock_boto3_client.get_object.side_effect = Exception("An error occured with boto3 client")

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        date_time_now = datetime(2025, 12, 30, 12, 0, 0)
        expected_s3_key = f"processed_data/processed_bikeshare_{date_time_now.strftime('%Y%m%d_%H%M%S')}.parquet"
        processed_data_key = f"processed_data/processed_bikeshare_{expected_s3_key}.parquet"

        with pytest.raises(Exception) as exc_info:
            process_result = loadtos3obj.process_rides_with_partitioning(
                processed_data_key=processed_data_key
            )
            
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while trying to deduplicate and load processed data to the transformed bucket"
            assert exc_info.value.args[0]['error'] == "An error occured with boto3 client"
    
    

    @patch('src.etl.impute_missing_station_ids')
    @patch('src.etl.clean_raw_data')
    @patch('src.etl.extract_source_data')
    def test_process_rides_with_partitioning_successful(
        self,
        mock_extract_source_data: MagicMock,
        mock_clean_raw_data: MagicMock,
        mock_impute_missing_station_ids: MagicMock,
        mock_boto3_client: BaseClient,
        upload_to_source_bucket: None,
        aws_credentials: Dict,
        raw_data_schema: Dict,
        valid_transformed_s3_config: Dict,
        valid_raw_s3_config: Dict,
        extract_and_valid_source_params: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.filter_duplicate_rides method successfully filters out duplicate rides based on existing metadata in the transformed S3 bucket.
        This test verifies that the method correctly identifies and filters out duplicate rides from the provided rides DataFrame.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            upload_metadata_to_transformed_bucket: Fixture to upload metadata to transformed bucket.
            aws_credentials: Mocked AWS credentials.
            raw_data_schema: Fixture providing the expected schema for the raw data.
        
        Returns:
            None
        """
        raw_df = pd.DataFrame({
                "ride_id": ["ride20", "ride21", "ride22", "ride23" ],
                "rideable_type": ["Electric_bike", "Docked_bike", "Classic_bike", "Electric_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00", "2023-04-03 10:00:00", "2023-04-03 11:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00", "2023-04-03 10:30:00", "2023-04-03 11:30:00"],
                "start_station_name": ["Start_station1", "start_station2", "start_station3", "start_station4"],
                "start_station_id": ["Start_Station_id1", "Start_sTation_id2", "Start_Station_id3", "Start_Station_id4"],
                "end_station_name": [np.nan, np.nan, np.nan, np.nan],
                "end_station_id": [np.nan, np.nan, np.nan, np.nan],
                "start_lat": ["41.8781", "41.8810", "41.8820", "41.8830"],
                "start_lng": ["-87.6298", "-87.6300", "-87.6310", "-87.6320"],
                "end_lat": ["41.8850", "41.8860", "41.8870", "41.8880"],
                "end_lng": ["-87.6340", "-87.6350", "-87.6360", "-87.6370"],
                "member_casual": ["casual", "member", "casual", "member"]
            })
        
        cleaned_df = pd.DataFrame(
            {
                "ride_id": ["ride20", "ride21", "ride22", "ride23"],
                "rideable_type": ["electric_bike", "docked_bike", "classic_bike", "electric_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00", "2023-04-03 10:00:00", "2023-04-03 11:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00", "2023-04-03 10:30:00", "2023-04-03 11:30:00"],
                "start_station_name": ["start_station1", "start_station2", "start_station3", "start_station4"],
                "start_station_id": ["start_station_id1", "start_station_id2", "start_station_id3", "start_station_id4"],
                "end_station_name": [np.nan, np.nan, np.nan, np.nan],
                "end_station_id": [np.nan, np.nan, np.nan, np.nan],
                "start_lat": ["41.8781", "41.8810", "41.8820", "41.8830"],
                "start_lng": ["-87.6298", "-87.6300", "-87.6310", "-87.6320"],
                "end_lat": ["41.8850", "41.8860", "41.8870", "41.8880"],
                "end_lng": ["-87.6340", "-87.6350", "-87.6360", "-87.6370"],
                "member_casual": ["casual", "member", "casual", "member"]
            }
        )

        imputed_df = pd.DataFrame(
            {
                "ride_id": ["ride20", "ride21", "ride22", "ride23"],
                "rideable_type": ["electric_bike", "docked_bike", "classic_bike", "electric_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00", "2023-04-03 10:00:00", "2023-04-03 11:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00", "2023-04-03 10:30:00", "2023-04-03 11:30:00"],
                "start_station_name": ["start_station1", "start_station2", "start_station3", "start_station4"],
                "start_station_id": ["start_station_id1", "start_station_id2", "start_station_id3", "start_station_id4"],
                "end_station_name": ["end_station1", "end_station2", "end_station3", "end_station4"],
                "end_station_id": ["end_station_id1", "end_station_id2", "end_station_id3", "end_station_id4"],
                "start_lat": ["41.8781", "41.8810", "41.8820", "41.8830"],
                "start_lng": ["-87.6298", "-87.6300", "-87.6310", "-87.6320"],
                "end_lat": ["41.8850", "41.8860", "41.8870", "41.8880"],
                "end_lng": ["-87.6340", "-87.6350", "-87.6360", "-87.6370"],
                "member_casual": ["casual", "member", "casual", "member"]
            }
        )

        source_bucket = valid_transformed_s3_config['source_bucket']
        source_s3_key = valid_transformed_s3_config['source_s3_key']


        upload_sample_to_source_bucket = upload_to_source_bucket(raw_df)

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}",
            "data": raw_df
        }

        mock_clean_raw_data.return_value = {
            "status": "success",
            "message": "Raw data cleaned successfully",
            "data": cleaned_df
        }

        mock_impute_missing_station_ids.return_value = {
            "status": "success",
            "message": "Missing station IDs imputed successfully",
            "data": imputed_df
        }

        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"

        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_raw_data = loadtos3obj.process_raw_data(
                extracted_dump_s3_key=valid_extracted_dump_s3_key,
                raw_data_schema=raw_data_schema
            )

            processed_data_key = process_raw_data['processed_data_key']
            process_and_upload_result = loadtos3obj.process_rides_with_partitioning(
                processed_data_key=processed_data_key
            )
            assert process_and_upload_result['status'] == "success"
            assert process_and_upload_result['message'] == "Successfully processed rides and uploaded with partitioning"
            assert process_and_upload_result['uploaded_records'] == 4  #new ride IDs: ride20, ride21, ride22, ride23, they have not been uploaded before
    

    @patch('src.etl.impute_missing_station_ids')
    @patch('src.etl.clean_raw_data')
    @patch('src.etl.extract_source_data')
    def test_process_rides_with_partitioning_deduplication(
        self,
        mock_extract_source_data: MagicMock,
        mock_clean_raw_data: MagicMock,
        mock_impute_missing_station_ids: MagicMock,
        mock_boto3_client: BaseClient,
        create_source_bucket: None,
        upload_to_source_bucket: None,
        aws_credentials: Dict,
        raw_data_schema: Dict,
        valid_transformed_s3_config: Dict,
        valid_raw_s3_config: Dict,
        extract_and_valid_source_params: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.process_rides_with_partitioning method successfully processes rides data and filters out duplicate rides based on existing metadata in the transformed S3 bucket.
        This test verifies that the method correctly identifies and filters out duplicate rides from the provided rides DataFrame.
        
        Args:
            mock_extract_source_data: A mocked extract_source_data function.
            mock_clean_raw_data: A mocked clean_raw_data function.
            mock_impute_missing_station_ids: A mocked impute_missing_station_ids function.
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_source_bucket: Fixture to create source bucket.
            upload_to_source_bucket: Fixture to upload sample data to source bucket.
            aws_credentials: Mocked AWS credentials.
            raw_data_schema: Fixture providing the expected schema for the raw data.
        
        Returns:
            None
        """

        test_duplicate_df = pd.DataFrame({
                "ride_id": ["ride21", "ride22", "ride24"],
                "rideable_type": ["Docked_bike", "Classic_bike", "Electric_bike"],
                "started_at": ["2023-04-03 09:00:00", "2023-04-03 10:00:00", "2023-04-03 11:00:00"],
                "ended_at": ["2023-04-03 09:30:00", "2023-04-03 10:30:00", "2023-04-03 11:30:00"],
                "start_station_name": ["start_station2", "start_station3", "start_station4"],
                "start_station_id": ["start_station_id2", "start_station_id3", "start_station_id4"],
                "end_station_name": ["end_station2", "end_station3", "end_station4"],
                "end_station_id": ["end_station_id2", "end_station_id3", "end_station_id4"],
                "start_lat": ["41.8810", "41.8820", "41.8830"],
                "start_lng": ["-87.6300", "-87.6310", "-87.6320"],
                "end_lat": ["41.8860", "41.8870", "41.8880"],
                "end_lng": ["-87.6350", "-87.6360", "-87.6370"],
                "member_casual": ["member", "casual", "member"]
            })
        
        cleaned_df = pd.DataFrame(
            {
                "ride_id": ["ride21", "ride22", "ride24"],
                "rideable_type": ["docked_bike", "classic_bike", "electric_bike"],
                "started_at": ["2023-04-03 09:00:00", "2023-04-03 10:00:00", "2023-04-03 11:00:00"],
                "ended_at": ["2023-04-03 09:30:00", "2023-04-03 10:30:00", "2023-04-03 11:30:00"],
                "start_station_name": ["start_station2", "start_station3", "start_station4"],
                "start_station_id": ["start_station_id2", "start_station_id3", "start_station_id4"],
                "end_station_name": ["end_station2", "end_station3", "end_station4"],
                "end_station_id": ["end_station_id2", "end_station_id3", "end_station_id4"],
                "start_lat": ["41.8810", "41.8820", "41.8830"],
                "start_lng": ["-87.6300", "-87.6310", "-87.6320"],
                "end_lat": ["41.8860", "41.8870", "41.8880"],
                "end_lng": ["-87.6350", "-87.6360", "-87.6370"],
                "member_casual": ["member", "casual", "member"]
            }
        )

        imputed_df = cleaned_df
        source_bucket = valid_transformed_s3_config['source_bucket']
        source_s3_key = valid_transformed_s3_config['source_s3_key']

        upload_sample_to_source_bucket = upload_to_source_bucket(test_duplicate_df)
        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}",
            "data": test_duplicate_df
        }
        mock_clean_raw_data.return_value = {
            "status": "success",
            "message": "Raw data cleaned successfully",
            "data": cleaned_df
        }
        mock_impute_missing_station_ids.return_value = {
            "status": "success",
            "message": "Missing station IDs imputed successfully",
            "data": imputed_df
        }

        valid_raw_bucket_weekly_dump_prefix = valid_raw_s3_config['raw_bucket_weekly_dump_prefix']
        valid_batch_year = extract_and_valid_source_params['batch_year']
        valid_batch_week = extract_and_valid_source_params['batch_week']
        valid_extracted_dump_s3_key = f"{valid_raw_bucket_weekly_dump_prefix}/trips_year_{valid_batch_year}_week_{valid_batch_week}.csv"


        loadtos3obj = LoadToTransformedS3(s3_config=valid_transformed_s3_config)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_raw_data = loadtos3obj.process_raw_data(
                extracted_dump_s3_key=valid_extracted_dump_s3_key,
                raw_data_schema=raw_data_schema
            )

            processed_data_key = process_raw_data['processed_data_key']
            process_and_upload_result = loadtos3obj.process_rides_with_partitioning(
                processed_data_key=processed_data_key
            )
            # expected_uploaded_ride_ids = {"ride24"} #ride21 and ride22 are duplicates (have been uploaded already based on existing metadata)
            assert process_and_upload_result['status'] == "success"
            assert process_and_upload_result['message'] == "Successfully processed rides and uploaded with partitioning"
            assert process_and_upload_result['uploaded_records'] == 1  #new ride ID: ride24 only


class TestLoadTransformedDataToSnowflake:
    """ 
    Test suite for LoadTransformedDataToSnowflake class methods.
    """

    def test_invalid_instantiation_args(self) -> None:
        
        """
        This test verifies that the LoadTransformedDataToSnowflake class raises the appropriate exceptions when instantiated with invalid arguments.
        The test checks for invalid types for the s3_config and snowflake_config parameters.

        Args:
            None

        Returns:
            None
        """
        invalid_snowflake_config = [
            "invalid_config", 
            123, 
            45.67, 
            None,
            [],
            set()
        ] #snowflake_config must be a dictionary

        for snowflake_config in invalid_snowflake_config:
            with pytest.raises(Exception) as exc_info:
                LoadTransformedDataToSnowflake(
                    snowflake_config=snowflake_config
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
            assert exc_info.value.args[0]['error'] == "snowflake_config must be a dictionary"
    
    def test_missing_snowflake_config_keys(self,
                                           valid_snowflake_config: dict) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class raises the appropriate exceptions when instantiated with missing required keys in the snowflake_config dictionary.

        Args:
            None

        Returns:
            None
        """

        required_keys = [
            'snowflake_account',
            'snowflake_username',
            'snowflake_password',
            'snowflake_warehouse',
            'snowflake_database',
            'snowflake_schema',
            'snowflake_role',
            'stage_name'
        ]
        missing_keys = ['snowflake_account','snowflake_username']
        missing_keys = sorted(missing_keys)
        valid_snowflake_config = valid_snowflake_config.copy()
        for key in missing_keys:
            valid_snowflake_config.pop(key)

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake(
                snowflake_config=valid_snowflake_config
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
        assert exc_info.value.args[0]['error'] == f"Missing required key in snowflake_config: {missing_keys}"

    def test_invalid_snowflake_config_values(self) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class raises the appropriate exceptions when instantiated with invalid snowflake_config values.
        The test checks for missing required keys in the snowflake_config dictionary.

        Args:
            None

        Returns:
            None
        """
        invalid_snowflake_configs = [
            (123, "snowflake_user", "snowflake_password",
              "snowflake_warehouse", "snowflake_database",
                "snowflake_schema", "snowflake_role", "snowflake_stage_name"),  # Invalid type for account
            
            ("snowflake_account", {}, "snowflake_password",
              "snowflake_warehouse", "snowflake_database",
                "snowflake_schema", "snowflake_role", "snowflake_stage_name"),  # invalid type for user
            
            ("snowflake_account", "snowflake_user", None,
              "snowflake_warehouse", "snowflake_database",
                "snowflake_schema", "snowflake_role", "snowflake_stage_name"),  #invalid type for password
            
            ("snowflake_account", "snowflake_user", "snowflake_password",
              pd.DataFrame(), "snowflake_database",
                "snowflake_schema", "snowflake_role", "snowflake_stage_name"),  # Invalid type for warehouse
            
            ("snowflake_account", "snowflake_user", "snowflake_password",
              "snowflake_warehouse", [],
                "snowflake_schema", "snowflake_role", "snowflake_stage_name"),  # Invalid type for database
            
            ("snowflake_account", "snowflake_user", "snowflake_password",
              "snowflake_warehouse", "snowflake_database",
                45.1, "snowflake_role", "snowflake_stage_name"),  # Invalid type for schema
            
            ("snowflake_account", "snowflake_user", "snowflake_password",
              "snowflake_warehouse", "snowflake_database",
                "snowflake_schema", set(), "snowflake_stage_name"),  # Invalid type for role
            
            ("snowflake_account", "snowflake_user", "snowflake_password",
              "snowflake_warehouse", "snowflake_database",
                "snowflake_schema", "snowflake_role", ()),  # Invalid type for stage name
        ]

        for snowflake_cred in invalid_snowflake_configs:
            snowflake_config = {
                    'snowflake_account': snowflake_cred[0],
                    'snowflake_username': snowflake_cred[1],
                    'snowflake_password': snowflake_cred[2],
                    'snowflake_warehouse': snowflake_cred[3],
                    'snowflake_database': snowflake_cred[4],
                    'snowflake_schema': snowflake_cred[5],
                    'snowflake_role': snowflake_cred[6],
                    'stage_name': snowflake_cred[7]
                }
            with pytest.raises(Exception) as exc_info:
                LoadTransformedDataToSnowflake(
                    snowflake_config=snowflake_config
                )

            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
            assert exc_info.value.args[0]['error'] == "All values in snowflake_config must be strings"
    

    def test_connection_string_successful(self,
                                            valid_snowflake_config: Dict
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class successfully creates a Snowflake connection string
        when instantiated with valid snowflake_config values.
        It also verifies that the generated connection string matches the expected format.
        It also verifies that create_engine method creates a SQLAlchemy engine instance successfully.

        Args:
            mock_create_engine: A mocked create_engine function.
            mock_sessionmaker: A mocked sessionmaker function.
            valid_snowflake_config: Fixture providing valid Snowflake configuration.

        Returns:
            None
        """

        load_snowflake_obj = LoadTransformedDataToSnowflake(
            snowflake_config=valid_snowflake_config
        )

        expected_conn_string = (
            "snowflake://test_user:test_password@test_account/"
            "test_database/test_schema?warehouse=test_warehouse&role=test_role"
        )

        assert load_snowflake_obj.connection_string == expected_conn_string
        # mock_create_engine.assert_called_once_with(expected_conn_string)
    

    @patch('src.etl.create_engine')
    def test_create_engine_instance_exception(
        self,
        mock_create_engine: MagicMock,
        valid_snowflake_config: Dict
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class handles exceptions raised during the creation of the SQLAlchemy engine instance.
        This test simulates an exception during the engine creation and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_create_engine: A mocked create_engine function.
            valid_snowflake_config: Fixture providing valid Snowflake configuration.

        Returns:
            None
        """
        mock_create_engine.side_effect = Exception("An error occurred while creating the SQLAlchemy engine instance")

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake(
                snowflake_config=valid_snowflake_config
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
        assert exc_info.value.args[0]['error'] == "An error occurred while creating the SQLAlchemy engine instance"
    
    @patch('src.etl.sessionmaker')
    def test_sessionmaker_instance_exception(
        self,
        mock_sessionmaker: MagicMock,
        valid_snowflake_config: Dict
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class handles exceptions raised during the creation of the sessionmaker instance.
        This test simulates an exception during the sessionmaker creation and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            valid_snowflake_config: Fixture providing valid Snowflake configuration.
            mock_sessionmaker: A mocked sessionmaker function.

        Returns:
            None
        """
        mock_sessionmaker.side_effect = Exception("An error occurred while creating the sessionmaker instance")

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake(
                snowflake_config=valid_snowflake_config
            )

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
        assert exc_info.value.args[0]['error'] == "An error occurred while creating the sessionmaker instance"


    @patch('src.etl.sessionmaker')
    @patch('src.etl.create_engine')
    def test_successful_initialization_creates_engine_and_session(
        self, 
        mock_create_engine: MagicMock,
        mock_sessionmaker: MagicMock,
        valid_snowflake_config: Dict
            
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake class successfully creates a SQLAlchemy engine and sessionmaker instance
        when instantiated with valid snowflake_config values.

        Args:
            mock_create_engine: A mocked create_engine function.
            mock_sessionmaker: A mocked sessionmaker function.
            valid_snowflake_config: Fixture providing valid Snowflake configuration.

        Returns:
            None
        """
        mock_engine_instance = MagicMock()
        mock_create_engine.return_value = mock_engine_instance

        mock_session_instance = MagicMock()
        mock_sessionmaker.return_value.return_value = mock_session_instance

        load_snowflake_obj = LoadTransformedDataToSnowflake(
            snowflake_config=valid_snowflake_config
        )

        mock_create_engine.assert_called_once_with(load_snowflake_obj.connection_string)
        mock_sessionmaker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine_instance)
        assert load_snowflake_obj.engine == mock_engine_instance
        assert load_snowflake_obj.session == mock_session_instance
    
    
    @patch('src.etl.create_session')
    def test_session_context_manager_exception(
        self,
        mock_create_session: MagicMock,
        loadertosnowflake: MagicMock
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake.load_from_stage_to_table method handles exceptions raised when creating the session context manager.
        This test simulates an exception during the session creation and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            loadertosnowflake: Fixture providing an instance of LoadTransformedDataToSnowflake.
        Returns:
            None
        """

        mock_create_session.side_effect = Exception("An error occurred while creating the session context manager")

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake.load_from_stage_to_table(loadertosnowflake)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while loading data from stage to table"
        assert exc_info.value.args[0]['error'] == "An error occurred while creating the session context manager"

    @patch('src.etl.create_session')
    def test_copy_loading_query_execution_failure(
        self,
        mock_create_session: MagicMock,
        loadertosnowflake: MagicMock
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake.load_from_stage_to_table method handles exceptions raised during the execution of the SQL query that copies data from the stage to the table.
        This test simulates an exception during the query execution and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_create_session: A mocked create_session function.
            loadertosnowflake: Fixture providing an instance of LoadTransformedDataToSnowflake.

        Returns:
            None
        """

        mock_db_session = MagicMock()
        mock_db_session.execute.side_effect = Exception("An error occurred during query execution")

        mock_create_session.return_value.__enter__.return_value = mock_db_session

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake.load_from_stage_to_table(loadertosnowflake)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while loading data from stage to table"
        assert exc_info.value.args[0]['error'] == "An error occurred during query execution"
    

    @patch('src.etl.create_session')
    def test_cluster_query_execution_failure(
        self,
        mock_create_session: MagicMock,
        loadertosnowflake: MagicMock
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake.load_from_stage_to_table method handles exceptions raised during the execution of the SQL query that adds clustering to the target table based on the appropriate fields.
        This test simulates an exception during the clustering query execution and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            loadertosnowflake: Fixture providing an instance of LoadTransformedDataToSnowflake.
        Returns:
            None
        """
        mock_db_session = MagicMock()
        mock_db_session.execute.side_effect = [None, Exception("An error occurred during clustering query execution")]
        mock_create_session.return_value.__enter__.return_value = mock_db_session
        
        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake.load_from_stage_to_table(loadertosnowflake)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while loading data from stage to table"
        assert exc_info.value.args[0]['error'] == "An error occurred during clustering query execution"

    @patch('src.etl.create_session')
    def test_commit_failure(
        self, 
        mock_create_session: MagicMock,
        loadertosnowflake: MagicMock
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake.load_from_stage_to_table method handles exceptions raised during the commit operation of the database session.
        This test simulates an exception during the commit and verifies that the method catches the exception and raises the appropriate error message.

        Args:
            mock_create_session: A mocked create_session function.
            loadertosnowflake: Fixture providing an instance of LoadTransformedDataToSnowflake
        Returns:
            None
        """
        mock_db_session = MagicMock()
        mock_db_session.commit.side_effect = Exception("An error occurred during commit operation")

        mock_create_session.return_value.__enter__.return_value = mock_db_session

        with pytest.raises(Exception) as exc_info:
            LoadTransformedDataToSnowflake.load_from_stage_to_table(loadertosnowflake)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while loading data from stage to table"
        assert exc_info.value.args[0]['error'] == "An error occurred during commit operation"
    
    @patch('src.etl.BikeRide')
    @patch('src.etl.create_session')
    def test_successful_loading_from_stage_to_table(
        self,
        mock_create_session: MagicMock,
        mock_bikeride: MagicMock,
        loadertosnowflake: MagicMock
    ) -> None:
        """
        This test verifies that the LoadTransformedDataToSnowflake.load_data_from_stage_to_table method successfully loads data from a Snowflake stage to a specified table.
        This test mocks the Snowflake session and verifies that the method executes the correct SQL command to load the data.

        Args:
            mock_create_session: A mocked create_session function.
            mock_bikeride: A mocked BikeRide model.
            loadertosnowflake: Fixture providing an instance of LoadTransformedDataToSnowflake.
            valid_snowflake_config: Fixture providing valid Snowflake configuration.
        
        Returns:
            None
        """
        mock_bikeride.__tablename__ = "raw_bike_rides"
        mock_bikeride.__table_args__ = {"schema": "RAW"}

        mock_db_session = MagicMock()
        mock_db_session.query.return_value.count.return_value = 2

        mock_create_session.return_value.__enter__.return_value = mock_db_session

        result = LoadTransformedDataToSnowflake.load_from_stage_to_table(loadertosnowflake)

        assert mock_db_session.execute.call_count == 2

        mock_db_session.commit.assert_called_once()

        mock_db_session.query.assert_called_once_with(mock_bikeride)

        # Return payload validation
        assert result["status"] == "success"
        assert result["no_of_new_records"] == 2
        assert result['message'] == f"Successfully loaded data from stage: {loadertosnowflake.stage_name} to table: {mock_bikeride.__tablename__}"
