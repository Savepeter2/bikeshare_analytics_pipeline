import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (gen_hash_key_station_id, extract_source_data,
                        validate_raw_data, requests_session_with_retries, get_address,
                        add_station_id, clean_raw_data, standardize_coordinates_and_fill_station_ids,
                        impute_missing_station_ids, validate_processed_data, send_slack_alert,
                        process_stream)

from typing import Dict, Callable, Generator
from unittest.mock import patch, MagicMock
from botocore.client import BaseClient
from boto3.session import Session
from pytest import MonkeyPatch
from requests.adapters import HTTPAdapter, Retry
from requests.sessions import Session
from requests.exceptions import ConnectionError
import random
import string

from moto import mock_aws
import boto3
import pandas as pd
import numpy as np
from io import BytesIO


@pytest.fixture(scope="class")
def aws_credentials():
    """
    These are FAKE credentials used only for mocking.
    They will never connect to real AWS.
    """
    return {
        'aws_access_key': 'AKIAIOSFODNN7EXAMPLE',
        'aws_secret_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
        'raw_bucket': "test-bucket",
        'raw_s3_key': "raw/raw_data.parquet",
        'transformed_bucket': "transformed-bucket",
        'transformed_s3_key': "transformed/staging_data.parquet",
        'source_bucket': "source-bucket",
        'source_s3_key': "source/source_data.csv"

    }


@pytest.fixture(scope="class")
def moto_boto3_client(aws_credentials: dict) -> Generator[BaseClient, None, None]:
    """
    Creates a mocked S3 client using moto. The real boto3 client is intercepted by moto.

    Args:
        aws_credentials: Fixture providing fake AWS credentials.

    Returns:
        A mocked boto3 S3 client.

    """
    with mock_aws():
        access_key = aws_credentials['aws_access_key']
        secret_key = aws_credentials['aws_secret_key']

        yield boto3.client('s3',
                           aws_access_key_id=access_key,
                           aws_secret_access_key=secret_key)

@pytest.fixture(scope="class")
def mock_boto3_client(aws_credentials: dict) -> BaseClient:
    """
    This fixture creates a mock boto3 client that simulates AWS interactions.
    It checks the provided AWS credentials against the expected mocked credentials used in creating the moto client.

    Args:
        aws_credentials: Fixture providing mocked AWS credentials.
    
    Returns:
        A mocked boto3 client function.
    """

    boto3_client = boto3.client

    def fake_client(service_name, **kwargs): 
        access_key = kwargs.get("aws_access_key_id")
        secret_key = kwargs.get("aws_secret_access_key")

        if access_key != aws_credentials['aws_access_key'] or \
           secret_key != aws_credentials['aws_secret_key']:
            raise Exception("The security token included in the request is invalid")

        return boto3_client(
            service_name,
            aws_access_key_id=aws_credentials['aws_access_key'],
            aws_secret_access_key=aws_credentials['aws_secret_key']
        )

    return fake_client

@pytest.fixture()
def mock_successful_session(monkeypatch) -> MagicMock:
    """
    This fixture mocks requests.Session to return a mocked session with retry configuration.
    """
    
    mocked_session = MagicMock(spec=Session)
    adapter = HTTPAdapter(max_retries=Retry(
        total=3,                
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
    ))

    mocked_session.adapters = {"https://": adapter}
    monkeypatch.setattr("src.utils.requests.Session", lambda: mocked_session)
    return mocked_session


@pytest.fixture
def mock_failing_session(monkeypatch):
    """Mocks requests.Session() to throw an exception."""
    def raise_failure(*args, **kwargs):
        raise Exception("boom!")
    
    monkeypatch.setattr("src.utils.requests.Session", raise_failure)


@pytest.fixture
def station_id_columns_dict() -> dict:
    """
    Fixture providing a dictionary mapping station ID columns to their corresponding station name columns.

    Args:
        None

    Returns:
        dict: A dictionary with station ID columns as keys and station name columns as values.
    """
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
    return station_id_columns_dict


@pytest.fixture(scope="class")
def create_transformed_bucket(moto_boto3_client: Session,
                              aws_credentials: dict) -> Dict:
    """
    This function sets up the mock transformed S3 bucket.

    Args:
        moto_boto3_client: Fixture providing a mocked boto3 client.
        aws_credentials: Fixture providing mocked AWS credentials.
    
    Returns:
        None
    """

    s3_config = {
                "transformed_bucket": aws_credentials['transformed_bucket'],
                "access_key": aws_credentials["aws_access_key"],
                "secret_key": aws_credentials["aws_secret_key"],
                "transformed_s3_key": aws_credentials['transformed_s3_key'],
                "region_name": "us-east-1"
            }

    bucket = s3_config['transformed_bucket']
    moto_boto3_client.create_bucket(Bucket=bucket)

    return s3_config

@pytest.fixture(scope="class")
def upload_transformed_data_to_s3(
                                    moto_boto3_client: Session,
                                    create_transformed_bucket: Dict,
                                ) -> None:
    """
    This function uploads a sample transformed parquet file to the mock transformed S3 bucket.
    Args:
        moto_boto3_client: Fixture providing a mocked boto3 client.
        aws_credentials: Fixture providing mocked AWS credentials.
    Returns:
        None
    """

    def _upload(df: pd.DataFrame) -> None:
        """
        This function uploads the provided DataFrame as a parquet file to the mock transformed S3 bucket.

        Args: 
            df (pd.DataFrame): The DataFrame to be uploaded as a parquet file.
        
        Returns:
            None
        """

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        s3_config = create_transformed_bucket
        bucket = s3_config['transformed_bucket']
        processed_s3_key = s3_config['transformed_s3_key']
        moto_boto3_client.put_object(
            Bucket=bucket,
            Key=processed_s3_key,
            Body=buffer.read()
        )
    return _upload


                   

class TestGenHashKeyStationID:
    def test_gen_hash_key_station_id_valid_input(self) -> None:
        """
        Test generating hash key for station ID with valid string inputs

        Args:
            None

        Returns:
            None
        """
        latitude = "40.7128"
        longitude = "-74.0060"
        result = gen_hash_key_station_id(latitude, longitude)
        expected_key = "8b3e36a5096917c815eebd8afe007427b4ffb0f2328e7d59f186959fc30f5f9d"
        expected_result = {
            "status": "success",
            "message": "Hash key generated successfully for station ID",
            "hash_key": expected_key
        }
        assert result == expected_result

    def test_gen_hash_key_station_id_invalid_input(self) -> None:
        """
        Test generating hash key for station ID with invalid latitude and longitude arguments where they are not strings

        Args:
            None

        Returns:
            None
        """
        invalid_inputs = [
            (123, "45.00"),       
            ("12.00", 45.00),    
            (None, "10.00"),
            ("10.00", None),
            ([], "10.00"),
            ("10.00", {}),
            (True, "10.00"),
            ("10.00", False),
            (40.7128, -74.0060)
        ]

        for latitude, longitude in invalid_inputs:
            with pytest.raises(Exception) as exc_info:
                gen_hash_key_station_id(latitude, longitude)

            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "Unable to generate hash key for station ID",
                "error": "Latitude and Longitude for generating station ID must be strings"
            }


class TestExtractSourceData:
    """
    Tests for extract_source_data function. It includes tests for valid and invalid inputs and all edge cases.

    Args:
        None

    Returns:
        None
    """
    def test_extract_data_invalid_access_key_type(self):
        """
        Tests the extract_source_data function with invalid AWS access key types.

        Returns:
            None
        """
        invalid_aws_argument_type = [
            (123, "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            (None, "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            ([], "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            ({}, "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            (True, "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            (False, "valid_secret_key", "test-bucket", "test-key", 10, 0, {}),
            (12.34, "valid_secret_key", "test-bucket", "test-key", 10, 0, {})
        ]
        for args in invalid_aws_argument_type:
            aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_size=batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
    def test_extract_data_invalid_secret_key_type(self):
        """
        Tests the extract_source_data function with invalid AWS secret key types.

        Returns:
            None
        """
        invalid_aws_argument_type = [
            ("valid_access_key", 123, "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", None, "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", [], "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", {}, "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", True, "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", False, "test-bucket", "test-key", 10, 0, {}),
            ("valid_access_key", 12.34, "test-bucket", "test-key", 10, 0, {})
        ]
        for args in invalid_aws_argument_type:
            aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_size=batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
    def test_extract_data_invalid_bucket_type(self):
        """
        Tests the extract_source_data function with invalid source bucket types.

        Returns:
            None
        """
        invalid_aws_argument_type = [
            ("valid_access_key", "valid_secret_key", 123, "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", None, "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", [], "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", {}, "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", True, "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", False, "test-key", 10, 0, {}),
            ("valid_access_key", "valid_secret_key", 12.34, "test-key", 10, 0, {})
        ]
        for args in invalid_aws_argument_type:
            aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_size=batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
#     def test_extract_data_invalid_s3_key_type(self):
#         """
#         Tests the extract_source_data function with invalid source s3 key types.

#         Returns:
#             None
#         """
#         invalid_aws_argument_type = [
#             ("valid_access_key", "valid_secret_key", "test-bucket", 123, 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", None, 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", [], 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", {}, 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", True, 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", False, 10, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", 12.34, 10, 0, {})
#         ]
#         for args in invalid_aws_argument_type:
#             aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
#             with pytest.raises(Exception) as exc_info:
#                 extract_source_data(
#                     aws_access_key=aws_access_key,
#                     aws_secret_key=aws_secret_key,
#                     source_bucket=source_bucket,
#                     source_s3_key=source_s3_key,
#                     batch_size=batch_size,
#                     last_row_index=last_row_index,
#                     raw_data_schema=raw_data_schema
#                 )
#             assert exc_info.type is Exception
#             assert exc_info.value.args[0] == {
#                 "status": "error",
#                 "message": "An error occurred while extracting data from source s3",
#                 "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
#             }
    
#     def test_extract_data_invalid_batch_size_type(self):
#         """
#         Tests the extract_source_data function with invalid batch size types.

#         Returns:
#             None
#         """
#         invalid_aws_argument_type = [
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", "ten", 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", None, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", [], 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", {}, 0, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 12.34, 0, {})
#         ]
#         for args in invalid_aws_argument_type:
#             aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
#             with pytest.raises(Exception) as exc_info:
#                 extract_source_data(
#                     aws_access_key=aws_access_key,
#                     aws_secret_key=aws_secret_key,
#                     source_bucket=source_bucket,
#                     source_s3_key=source_s3_key,
#                     batch_size=batch_size,
#                     last_row_index=last_row_index,
#                     raw_data_schema=raw_data_schema
#                 )
#             assert exc_info.type is Exception
#             assert exc_info.value.args[0] == {
#                 "status": "error",
#                 "message": "An error occurred while extracting data from source s3",
#                 "error": "Batch size must be an integer"
#             }
    
#     def test_extract_data_invalid_last_row_index_type(self):
#         """
#         Tests the extract_source_data function with invalid last row index types.
#         Returns:
#             None
#         """
#         invalid_aws_argument_type = [
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 10, "zero", {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 20, None, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 30, [], {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 40, {}, {}),
#             ("valid_access_key", "valid_secret_key", "test-bucket", "test-key", 50, 12.34, {})
#         ]
#         for args in invalid_aws_argument_type:
#             aws_access_key, aws_secret_key, source_bucket, source_s3_key, batch_size, last_row_index, raw_data_schema = args
#             with pytest.raises(Exception) as exc_info:
#                 extract_source_data(
#                     aws_access_key=aws_access_key,
#                     aws_secret_key=aws_secret_key,
#                     source_bucket=source_bucket,
#                     source_s3_key=source_s3_key,
#                     batch_size=batch_size,
#                     last_row_index=last_row_index,
#                     raw_data_schema=raw_data_schema
#                 )
#             assert exc_info.type is Exception
#             assert exc_info.value.args[0] == {
#                 "status": "error",
#                 "message": "An error occurred while extracting data from source s3",
#                 "error": "Last row index must be an integer"
#             }

#     def test_load_raw_bucket(self, moto_boto3_client: Session,
#                             aws_credentials: dict) -> None:
#         """
#         This function sets up the mock raw S3 bucket and uploads a sample CSV file to it.

#         Args:
#             moto_boto3_client: Fixture providing a mocked boto3 client.
#             aws_credentials: Fixture providing mocked AWS credentials.
        
#         Returns:
#             None
#         """  
    
#         bucket = aws_credentials['raw_bucket']
#         s3_key = aws_credentials['raw_s3_key']
#         moto_boto3_client.create_bucket(Bucket=bucket)
#         moto_boto3_client.put_object(
#         Bucket=bucket,
#         Key=s3_key,
#         #7 rows pd.DataFrame({'col1':[1,3,5,7,9,11,13],'col2':[2,4,6,8,10,12,14]})
#         Body = b"col1,col2\n3,4\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14" 
#     )
        
#     def test_extract_data_unauthorized_credentials(self,
#                                         mock_boto3_client: Session,
#                                         aws_credentials: dict) -> None:
        
#         """
#         Tests the extract_source_data function with unauthorized AWS credentials.
#         It uses moto to mock S3 interactions.

#         Args:
#             mock_boto3_client: Fixture providing a mocked boto3 client.
#             aws_credentials: Fixture providing mocked AWS credentials.

#         Returns:
#             None
#         """
#         bucket = aws_credentials['raw_bucket']
#         s3_key = aws_credentials['raw_s3_key']

#         invalid_aws_credentials = [
#             ("INVALIDACCESSKEY", aws_credentials['aws_secret_key']),
#             (aws_credentials['aws_access_key'], "INVALIDSECRETKEY"),
#             ("INVALIDACCESSKEY", "INVALIDSECRETKEY"),
#         ]

#         for aws_access_key, aws_secret_key in invalid_aws_credentials:
#             with patch('boto3.client', side_effect=mock_boto3_client):
#                 with pytest.raises(Exception) as exc_info:
#                     extract_source_data(
#                         aws_access_key=aws_access_key,
#                         aws_secret_key=aws_secret_key, 
#                         source_bucket=bucket,
#                         source_s3_key=s3_key,
#                         batch_size=10,
#                         last_row_index=0,
#                         raw_data_schema={}
#                     )
#                 assert exc_info.type is Exception
#                 assert exc_info.value.args[0]["status"] == "error"
#                 assert "The security token included in the request is invalid" in exc_info.value.args[0]["error"]


#     def test_extract_data_first_batch(self, 
#                                       mock_boto3_client: Session,
#                                       aws_credentials: dict) -> None:
#         """
#         Tests the extract_source_data function with valid inputs.
#         It uses moto to mock S3 interactions.
#         It tests the first batch extraction with a batch size of 3 and starting from row index 0.
        
#         Args:
#             mock_s3_client: Fixture providing a mocked S3 client.
        
#         Returns:
#             None
#         """
#         bucket = aws_credentials['raw_bucket']
#         s3_key = aws_credentials['raw_s3_key']
#         batch_size = 3 #assuming batch size of 3 - will only extract 3 rows
#         last_row_index = 0 #starting from the beginning
        
#         with patch('boto3.client', side_effect=mock_boto3_client):
#             result = extract_source_data(
#                 aws_access_key=aws_credentials['aws_access_key'],
#                 aws_secret_key=aws_credentials['aws_secret_key'], 
#                 source_bucket=bucket,
#                 source_s3_key=s3_key,
#                 batch_size=batch_size,
#                 last_row_index=last_row_index,
#                 raw_data_schema={}
#             )

#             assert result["status"] == "success"
#             assert result["message"] == \
#             f"Successfully extracted data from source s3 bucket: {bucket}, s3 key: {s3_key}, rows: {last_row_index + 1} to {batch_size + last_row_index}"
#             pd.testing.assert_frame_equal(result["data"].reset_index(drop=True), 
#                                         pd.DataFrame({'col1':[1,3,5],'col2':[2,4,6]}).\
#                                             reset_index(drop=True))

#     def test_extract_data_second_batch(self, 
#                                        aws_credentials: dict,
#                                        mock_boto3_client: Session) -> None:
#         """
#         Tests the extract_source_data function with valid inputs.
#         It uses moto to mock S3 interactions.
#         It tests the second batch extraction with a batch size of 3 and starting from row index 3.
#         Args:
#             aws_credentials: Fixture providing fake AWS credentials.
#             mock_s3_client: Fixture providing a mocked S3 client.
#         Returns:
#             None
#         """
#         bucket = aws_credentials['raw_bucket']
#         s3_key = aws_credentials['raw_s3_key']
#         batch_size = 3 #assuming batch size of 3 - will only extract 3 rows
#         last_row_index = 3 #starting from row index 3(second batch)
        
#         with patch('boto3.client', side_effect=mock_boto3_client):
#             result = extract_source_data(
#                 aws_access_key=aws_credentials['aws_access_key'],
#                 aws_secret_key=aws_credentials['aws_secret_key'], 
#                 source_bucket=bucket,
#                 source_s3_key=s3_key,
#                 batch_size=batch_size,
#                 last_row_index=last_row_index,
#                 raw_data_schema={}
#             )
            
#             assert result["status"] == "success"
#             assert result["message"] == \
#             f"Successfully extracted data from source s3 bucket: {bucket}, s3 key: {s3_key}, rows: {last_row_index + 1} to {batch_size + last_row_index}"
#             pd.testing.assert_frame_equal(result["data"].reset_index(drop=True), 
#                                         pd.DataFrame({'col1':[7,9,11],'col2':[8,10,12]}).\
#                                             reset_index(drop=True))
 


# class TestValidateRawData:
#     def test_invalid_datatype_argument(self):
#         """
#         Tests the validate_raw_data function with an invalid argument type.
#         i.e argument is not a dataframe

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_raw_df_types = [
#             123,
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34 
#         ]
#         for raw_df in invalid_raw_df_types:
#             with pytest.raises(Exception) as exc_info:
#                 validate_raw_data(raw_df)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"
    
#     def test_missing_required_columns(self):
#         """
#         Tests the validate_raw_data function with a DataFrame missing required columns.

#         It should raise an Exception indicating which columns are missing.
#         Args:
#             None
#         Returns:
#             None
#         """
#         raw_df = pd.DataFrame({
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_lat": [40.7128, 34.0522, 41.8781],
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, -87.6288],
#             "member_casual": ["member", "casual", "member"]
#         })
#         missing_columns = ['ride_id', 'rideable_type', 'started_at',
#                            'start_station_id', 'start_station_name',
#                            'end_station_id', 'end_station_name']
        
        
#         with pytest.raises(Exception) as exc_info:
            
#             validate_raw_data(raw_df)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column validation error occured",
#             "error": f"columns: {sorted(missing_columns)} are missing"
#         }
    
#     def test_identifier_column_null(self):
#         """
#         Tests the validate_raw_data function with a DataFrame having null values in the identifier column 'ride_id'.

#         It will raise an Exception indicating that 'ride_id' contains null values.
        
#         Args:
#             None
        
#         Returns:
#             None
#         """
#         test_df = pd.DataFrame({
#             "ride_id": [None, "ride_002", "ride_003"],
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
#             "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_station_name": ["Station A", "Station B", "Station C"],
#             "end_station_name": ["Station D", "Station E", "Station F"],
#             "start_station_id": ["station_id1", "station_id2", "station_id3"],
#             "end_station_id": ["station_id4", "station_id5", "station_id6"],
#             "start_lat": [40.7128, 34.0522, 41.8781],
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, -87.6288],
#             "member_casual": ["member", "casual", "member"]
#         })
        
#         with pytest.raises(Exception) as exc_info:
#             validate_raw_data(test_df)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column validation error occured",
#             "error": "ride_id column contains null values"
#         }
    
#     def test_identifier_column_not_unique(self):
#         """
#         Tests the validate_raw_data function with a DataFrame having duplicate values in the identifier column 'ride_id'.

#         It will raise an Exception indicating that 'ride_id' contains duplicate values.
        
#         Args:
#             None

#         Returns:
#             None
#         """
#         test_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", "ride_001"],
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
#             "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_station_name": ["Station A", "Station B", "Station C"],
#             "end_station_name": ["Station D", "Station E", "Station F"],
#             "start_station_id": ["station_id1", "station_id2", "station_id3"],
#             "end_station_id": ["station_id4", "station_id5", "station_id6"],
#             "start_lat": [40.7128, 34.0522, 41.8781],
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, -87.6288],
#             "member_casual": ["member", "casual", "member"]
#         })
        
#         with pytest.raises(Exception) as exc_info:
#             validate_raw_data(test_df)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column validation error occured",
#             "error": "ride_id column contains duplicate values"
#         }
    
#     def test_incorrect_datetime_format(self):
#         """
#         Tests the validate_raw_data function with a DataFrame having incorrect datetime format in 'started_at' column.

#         It will raise an Exception indicating that 'started_at' column has incorrect datetime format.
        
#         Args:
#             None

#         Returns:
#             None
#         """
#         test_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", "ride_003"],
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
#             "started_at": ["2023/01/01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"], #incorrect format in first row
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_station_name": ["Station A", "Station B", "Station C"],
#             "end_station_name": ["Station D", "Station E", "Station F"],
#             "start_station_id": ["station_id1", "station_id2", "station_id3"],
#             "end_station_id": ["station_id4", "station_id5", "station_id6"],
#             "start_lat": [40.7128, 34.0522, 41.8781],
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, -87.6288],
#             "member_casual": ["member", "casual", "member"]
#         })

#         failing_datetime_columns = ['started_at']
#         with pytest.raises(Exception) as exc_info:
#             validate_raw_data(test_df)
            
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column type validation error(s) occurred",
#             "error": f"columns with invalid datetime format: {', '.join(failing_datetime_columns)}"
#         }
    
#     def test_str_column_with_non_str_values(self) -> None:
#         """
#         Tests the validate_raw_data function with a DataFrame having non-string values in a string column.

#         It will raise an Exception indicating that the specific string column contains non-string values.
        
#         Args:
#             None

#         Returns:
#             None
#         """
#         test_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", 12345], #non-string value in third row
#             "rideable_type": ["electric_bike", "docked_bike", 12345], #non-string value in third row
#             "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_station_name": ["Station A", "Station B", "Station C"],
#             "end_station_name": ["Station D", "Station E", "Station F"],
#             "start_station_id": ["station_id1", "station_id2", "station_id3"],
#             "end_station_id": ["station_id4", "station_id5", "station_id6"],
#             "start_lat": [40.7128, 34.0522, 41.8781],
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, -87.6288],
#             "member_casual": ["member", "casual", "member"]
#         })

#         failing_cols = ["ride_id", "rideable_type"]
        
#         with pytest.raises(Exception) as exc_info:
#             validate_raw_data(test_df)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column type validation error(s) occurred",
#             "error": f"columns with non-string values: {', '.join(failing_cols)}"
#         }
#     def test_float_column_with_non_float_values(self) -> None:
#         """
#         Tests the validate_raw_data function with a DataFrame having non-float values in a float column.

#         It will raise an Exception indicating that the specific float column contains non-float values.
        
#         Args:
#             None

#         Returns:
#             None
#         """
#         test_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", "ride_003"],
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
#             "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
#             "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
#             "start_station_name": ["Station A", "Station B", "Station C"],
#             "end_station_name": ["Station D", "Station E", "Station F"],
#             "start_station_id": ["station_id1", "station_id2", "station_id3"],
#             "end_station_id": ["station_id4", "station_id5", "station_id6"],
#             "start_lat": [40.7128, "invalid_latitude", 41.8781], #non-float value in second row
#             "start_lng": [-74.0060, -118.2437, -87.6298],
#             "end_lat": [40.7138, 34.0532, 41.8791],
#             "end_lng": [-74.0050, -118.2427, "invalid_longitude"], #non-float value in third row
#             "member_casual": ["member", "casual", "member"]
#         })

#         failing_cols = ["start_lat", "end_lng"]
        
#         with pytest.raises(Exception) as exc_info:
#             validate_raw_data(test_df)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
#         assert exc_info.value.args[0]["error"] == {
#             "status": "error",
#             "message": "column type validation error(s) occurred",
#             "error": f"columns with non-float values: {', '.join(failing_cols)}"
#         }


# def test_requests_session_success(mock_successful_session: MagicMock) -> None:
#     """
#     Tests the requests_session_with_retries function to ensure it configures
#     a requests Session with retry logic.

#     Args:
#         mock_successful_session: Fixture providing a mocked requests Session.

#     Returns:
#         None
#     """
#     session = requests_session_with_retries()

#     assert isinstance(session, Session)

#     print("session:", session)

#     assert session is mock_successful_session

#     adapter = session.adapters["https://"]
#     retries = adapter.max_retries

#     assert retries.total == 3
#     assert retries.backoff_factor == 2
#     assert retries.status_forcelist == [429, 500, 502, 503, 504]
#     assert retries.allowed_methods == ["GET", "POST"]


# def test_requests_session_failure(mock_failing_session: MagicMock) -> None:
#     """
#     Tests the requests_session_with_retries function to ensure it raises an exception
#     when the requests Session creation fails.

#     Args:
#         mock_failing_session: Fixture providing a mocked requests Session that raises an exception.

#     Returns:
#         None
#     """
#     with pytest.raises(Exception) as exc_info:
#         requests_session_with_retries()

#     mocked_error = exc_info.value.args[0]

#     assert mocked_error["status"] == "error"
#     assert mocked_error["message"] == "An error occurred while creating requests session with retries"
#     assert mocked_error["error"] == "boom!"

# class TestGetAddress:

#     def test_invalid_argument_types(self) -> None:
#         """
#         Tests the get_address function with invalid argument types.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_inputs = [
#             (123, -74.0060),       
#             ("40.7128", -74.0060),    
#             (None, "-74.0060"),
#             ("40.7128", None),
#             ([], -74.0060),
#             ("40.7128", {}),
#             (True, -74.0060),
#             ("40.7128", False),
#             (40.7128, "-74.0060")
#         ]

#         for latitude, longitude in invalid_inputs:
#             with pytest.raises(Exception) as exc_info:
#                 get_address(latitude, longitude)

#             assert exc_info.type is Exception
#             assert exc_info.value.args[0] == {
#                 "status": "error",
#                 "message": "An error occurred while fetching address from OpenStreetMap API",
#                 "error": "Latitude and Longitude must be string types"
#             }

#     @patch("src.utils.requests_session_with_retries")
#     def test_success_case(self, mock_factory) -> None:
#         """
#         Test that get_address returns a successful response when the API
#         returns HTTP 200 with a valid address payload.

#         This test simulates:
#             - A normal API request returning status code 200.
#             - A JSON body containing a display_name field.

#         Args:
#             mock_factory (MagicMock): The patched requests_session_with_retries function
#                 that returns a mock session whose GET request simulates a successful API response.
        
#         Returns:
#             None
#         """

#         mock_session = MagicMock()
#         mock_response = MagicMock()
#         mock_response.status_code = 200
#         mock_response.json.return_value = {
#             "address": {
#                 "city": "New York",
#                 "state": "NY",
#                 "country": "USA"
#             }
#         }
#         mock_session.get.return_value = mock_response
#         # Make the factory return this session
#         mock_factory.return_value = mock_session
#         latitude = "40.7128"
#         longitude = "-74.0060"
#         result = get_address(latitude, longitude)

#         assert result == {
#             "status": "success",
#             "message": "Address fetched from OpenStreetMap API successfully",
#             "data": "Address not found"
#         }


#     @patch("src.utils.requests_session_with_retries")
#     def test_429_rate_limit(self, mock_factory: MagicMock) -> None:
#         """
#         Test that get_address raises an exception when the API returns
#         HTTP 429 (rate limit exceeded).

#         This test simulates:
#             - A single API request returning status code 429.
#             - The function raising an exception with the corresponding error payload.

#         Args:
#             mock_factory (MagicMock): The patched requests_session_with_retries function
#                 that returns a mock session whose GET request simulates a 429 rate limit response.
#         """

#         mock_session = MagicMock()
#         mock_response = MagicMock()
#         mock_response.status_code = 429
#         mock_response.text = "Rate limit exceeded"
#         mock_session.get.return_value = mock_response

#         # Make the factory return this session
#         mock_factory.return_value = mock_session

#         latitude = "40.7128"
#         longitude = "-74.0060"

#         with pytest.raises(Exception) as exc_info:
#             result = get_address(latitude, longitude)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while fetching address from OpenStreetMap API"
#         assert exc_info.value.args[0]['error'] == {
#             "status": "error",
#             "message": "Rate limit exceeded when fetching data from OpenStreetMap API",
#             "error": f"{mock_response.text}"
#         }

#     @patch("src.utils.requests_session_with_retries")
#     def test_connection_error_retry(self, mock_factory: MagicMock, 
#                                     monkeypatch: MonkeyPatch) -> None:
#         """
#         Test that get_address retries once when a ConnectionError occurs and then succeeds.

#         This test simulates:
#             - The first request raising a ConnectionError.
#             - The second request returning a successful response.
#             - time.sleep being patched to avoid slowing down tests.

#         Args:
#             mock_factory (MagicMock): The patched requests_session_with_retries function that returns a mock session.
#             monkeypatch (MonkeyPatch): Used to patch time.sleep to skip actual waiting.
#         """

#         mock_response = MagicMock()
#         mock_session = MagicMock()
#         mock_response.status_code = 200
#         mock_response.json.return_value = {
#             "mocked_address": "data"
#         }
#         mock_session.get.side_effect = [
#         ConnectionError("Connection timed out"),
#         mock_response
#         ]
#         mock_session.get.return_value = mock_response

#         mock_factory.return_value = mock_session

#         # Patch sleep to avoid slowing down test
#         monkeypatch.setattr("src.utils.time.sleep", lambda x: None)

#         resp = get_address("10", "20")
#         assert resp["status"] == "success"
#         assert resp["message"] == "Address fetched from OpenStreetMap API successfully"
#         assert resp["data"] == "Address not found"
#         assert mock_session.get.call_count == 2

#     @patch("src.utils.requests_session_with_retries")
#     def test_connection_error_then_failure(self, mock_factory: MagicMock, monkeypatch: MonkeyPatch) -> None:
#         """
#         Test that get_address retries once after a ConnectionError and raises an exception
#         if the second attempt also fails.

#         This test simulates:
#             - The first request raising a ConnectionError.
#             - The retry attempt returning a non-200 failure response.
#             - time.sleep being patched to prevent delays during testing.

#         Args:
#             mock_factory (MagicMock): The patched requests_session_with_retries function
#                 used to supply mock session objects for each request attempt.
#             monkeypatch (MonkeyPatch): Pytest fixture used to patch time.sleep so the retry
#                 does not pause the test.
#         """

#         mock_session = MagicMock()
#         mock_response = MagicMock()
#         mock_response.status_code = 500
#         mock_response.text = "Internal Server Error"
#         mock_session.get.side_effect = [
#             ConnectionError("Connection timed out"),
#             mock_response
#         ] 
#         mock_factory.return_value = mock_session

#         monkeypatch.setattr("src.utils.time.sleep", lambda x: None)

#         with pytest.raises(Exception) as exc_info:
#             get_address("10", "20")

#         assert mock_session.get.call_count == 2
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while making request to OpenStreetMap API"
#         assert exc_info.value.args[0]['error'] == f"{mock_response.text}"

#     @patch("src.utils.requests_session_with_retries")
#     def test_non_200_non_429_response(self, mock_factory: MagicMock) -> None:
#         """
#         Test that get_address raises an exception when the API returns
#         a non-200 and non-429 HTTP status code (e.g., 500 server error).

#         This test simulates:
#             - A single request returning a failure response such as HTTP 500.
#             - No retry, since only ConnectionError triggers retry logic.

#         Args:
#             mock_factory (MagicMock): The patched requests_session_with_retries function
#                 used to return a mock session whose GET request simulates a server error.
#         """
#         mock_session = MagicMock()
#         mock_response = MagicMock()
#         mock_response.status_code = 500
#         mock_response.text = "Internal Server Error"
#         mock_session.get.return_value = mock_response

#         latitude = "40.7128"
#         longitude = "-74.0060"

#         mock_factory.return_value = mock_session

#         with pytest.raises(Exception) as exc_info:
#             get_address(latitude, longitude)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while fetching address from OpenStreetMap API"
#         assert exc_info.value.args[0]['error'] == {
#             "status": "error",
#             "message": "An error occurred while making request to OpenStreetMap API",
#             "error": f"{mock_response.text}"
#         }

# class TestAddStationId:

#     def test_add_station_id_invalid_argument_types(self) -> None:
#         """
#         Tests the add_station_id function with invalid argument types.

#         Args:
#             mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

#         Returns:
#             None
#         """
#         invalid_inputs = [
#             (123, "-74.0060"),
#             ("40.7128", -74.0060),    
#             (None, "-74.0060"),
#             ("40.7128", None),
#             (40.7128, "-74.0060"),
#             ("40.712", {}),
#             ([], -74.0060),
#             (True, -74.0060),
#         ]

#         for lat, lon in invalid_inputs:
#             with pytest.raises(Exception) as exc_info:
#                 add_station_id(lat, lon)

#             assert exc_info.type is Exception
#             assert exc_info.value.args[0]['status'] == "error"
#             assert exc_info.value.args[0]['message'] == "An error occurred while generating station ID"
#             assert exc_info.value.args[0]['error'] == "Latitude and Longitude must be string types"

#     @patch("src.utils.gen_hash_key_station_id")
#     def test_add_station_id_success(self, 
#                                     mock_gen_hash_key: MagicMock) -> None:
#         """
#         Tests the add_station_id function with valid inputs.
#         It verifies that the function returns the expected station ID.

#         Args:
#             mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

#         Returns:
#             None
#         """
#         latitude = "40.7128"
#         longitude = "-74.0060"
#         expected_hash_key_response = {
#             "status": "success",
#             "message": "Hash key generated successfully",
#             "hash_key": "station_12345"
#         }

#         mock_gen_hash_key.return_value = expected_hash_key_response
#         expected_station_id = mock_gen_hash_key.return_value['hash_key']

#         result = add_station_id(latitude, longitude)

#         assert result['status'] == "success"
#         assert result['message'] == "Station ID generated successfully"
#         assert result['data'] == expected_station_id
#         mock_gen_hash_key.assert_called_once_with(latitude, longitude)

#     @patch("src.utils.gen_hash_key_station_id")
#     def test_add_station_id_gen_hash_key_failure(self, 
#                                                  mock_gen_hash_key: MagicMock) -> None:
#         """
#         Tests the add_station_id function when gen_hash_key_station_id fails.
#         It verifies that the function raises an exception with the expected error message.

#         Args:
#             mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

#         Returns:
#             None
#         """
#         latitude = "40.7128"
#         longitude = "-74.0060"

#         mocked_gen_hash_key_error = {
#             "status": "error",
#             "message": "Unable to generate hash key for station ID",
#             "error": "hash key generation error"
#         }

#         mock_gen_hash_key.side_effect = Exception(
#             mocked_gen_hash_key_error
#         )
#         expected_error_response = {
#             "status": "error",
#             "message": "An error occurred while generating station ID",
#             "error": mocked_gen_hash_key_error
#         }

#         with pytest.raises(Exception) as exc_info:
#             add_station_id(latitude, longitude)

#         assert exc_info.type is Exception
#         assert exc_info.value.args[0] == expected_error_response
#         mock_gen_hash_key.assert_called_once_with(latitude, longitude)


# class TestCleanRawData:
#     """
#     Tests for the clean_raw_data function.
#     """

#     def test_invalid_argument_type(self) -> None:
#         """
#         Tests the clean_raw_data function with an invalid argument type.
#         i.e argument is not a dataframe

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_raw_df_types = [
#             "invalid_df",
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             123,
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]

#         for raw_df in invalid_raw_df_types:
#             with pytest.raises(Exception) as exc_info:
#                 clean_raw_data(raw_df)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred while cleaning raw data"
#         assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"
    
#     def test_deduplication_and_standardization(self) -> None:
#         """
#         Tests the clean_raw_data function for deduplication and standardization of column names(all lowercase).
#         It tests the success case where the input DataFrame contains duplicate rows and string type columns with inconsistent casing and leading/trailing spaces.
#         It verifies that duplicate rows based on 'ride_id' are removed and string type columns are stripped and converted to lowercase.

#         Args:
#             None
        
#         Returns:
#             None
#         """
#         raw_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", "ride_001", "ride_003"], #duplicate ride_id "ride_001"
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike", "electric_bike"],
#             "start_station_name": [" Station A ", "Station B", " station a ", "Station C"],
#             "start_station_id": ["ST001", "ST002", "ST001", "ST003"],
#             "end_station_name": [" Station X ", "Station Y", " station x ", "Station Z"],
#             "end_station_id": ["STX01", "STY02", "STX01", "STZ03"],
#             "member_casual": ["Member", "Casual", "Member", "Member"]
#         })

#         expected_cleaned_df = pd.DataFrame({
#             "ride_id": ["ride_001", "ride_002", "ride_003"],
#             "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
#             "start_station_name": ["station a", "station b", "station c"],
#             "start_station_id": ["st001", "st002", "st003"],
#             "end_station_name": ["station x", "station y", "station z"],
#             "end_station_id": ["stx01", "sty02", "stz03"],
#             "member_casual": ["member", "casual", "member"]
#         }).reset_index(drop=True)

#         result = clean_raw_data(raw_df)
#         df_result = result['data'].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_result,
#                                        expected_cleaned_df)
#         assert df_result.duplicated().sum() == 0
#         assert df_result.columns.str.islower().all()
#         assert (df_result.columns == df_result.columns.str.strip()).all()


# class TestImputeMissingStationIDs:
#     """
#     Tests for the impute_missing_station_ids function.
#     """ 
#     def test_invalid_argument_type(self) -> None:
#         """
#         Tests the impute_missing_station_ids function with an invalid argument type.
#         i.e argument is not a dataframe

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_raw_df_types = [
#             "invalid_df",
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             123,
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]

#         for raw_df in invalid_raw_df_types:
#             with pytest.raises(Exception) as exc_info:
#                 impute_missing_station_ids(raw_df)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]["status"] == "error"
#         assert exc_info.value.args[0]["message"] == "An error occurred while imputing missing station IDs"
#         assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"


class TestStandardizeCoordinatesAndFillStationIds:
    """
    Tests for the standardize_coordinates_and_fill_station_ids function.
    """ 
    def test_invalid_argument_type(self) -> None:
        """
        Tests the standardize_coordinates_and_fill_station_ids function with an invalid argument type.
        i.e argument is not a dataframe

        Args:
            None

        Returns:
            None
        """
        invalid_record_types = [ 
            "invalid_df",
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            123,
            None,
            [],
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]

        station_id_columns_dict ={
            "start_station_id": "start_latitude,start_longitude",
            "end_station_id": "end_latitude,end_longitude"
        }

        for record_type in invalid_record_types:
            with pytest.raises(Exception) as exc_info:
                standardize_coordinates_and_fill_station_ids(record_type, station_id_columns_dict) #record_type must be a dataframe
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while standardizing coordinates and filling station IDs"
        assert exc_info.value.args[0]["error"] == "Input record must be a dictionary"
    
    def test_invalid_station_id_columns_dict_type(self) -> None:
        """
        Tests the standardize_coordinates_and_fill_station_ids function with an invalid station_id_columns_dict type.
        i.e station_id_columns_dict is not a dictionary
        
        Args:
            None

        Returns:
            None
        """
        invalid_station_id_columns_dict_types = [ 
            "invalid_dict",
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            123,
            None,
            [],
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        record = {
            "start_latitude": "40.7128",
            "start_longitude": "-74.0060",
            "end_latitude": "40.7138",
            "end_longitude": "-74.0050"
        }

        for station_id_columns_dict in invalid_station_id_columns_dict_types:
            with pytest.raises(Exception) as exc_info:
                standardize_coordinates_and_fill_station_ids(record, station_id_columns_dict) #station_id_columns_dict must be a dictionary
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while standardizing coordinates and filling station IDs"
        assert exc_info.value.args[0]["error"] == "Input station_id_columns_dict must be a dictionary"
    

    @patch("src.utils.get_address") 
    @patch("src.utils.add_station_id")
    def test_empty_station_name_and_id(self,
                                    mock_add_station_id: MagicMock,
                                   mock_get_address: MagicMock
                                   ) -> None:
        """
        Tests the standardize_coordinates_and_fill_station_ids function when station names and IDs are empty.
        It verifies that the function generates hashed station IDs using coordinates and fetches the corresponding addresses.
        
        Args:
            mock_add_station_id (MagicMock): The patched add_station_id function.
            mock_get_address (MagicMock): The patched get_address function.

        Returns:
            None
        """
        cleaned_df = pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE456"],
            "rideable_type": ["electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", np.nan],
            "start_station_id": ["Station_id1", np.nan],
            "end_station_name": ["Station2", np.nan],
            "end_station_id": ["Station_id2", np.nan],
            "start_lat": ["40.7128", "34.0522"],
            "start_lng": ["-74.0060", "-118.2437"],
            "end_lat": ["40.7138", "34.0532"],
            "end_lng": ["-74.0050", "-118.2427"],
            "member_casual": ["member", "casual"]
        })
        
        mock_get_address.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": f"Address for ({lat}, {lng})"
        }
    )
        
        mock_add_station_id.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": f"hashed_station_{lat}_{lng}"
        }
    )
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
        cleaned_records = cleaned_df.to_dict(orient="records")
        for record in cleaned_records:
            result = standardize_coordinates_and_fill_station_ids(record, station_id_columns_dict)
            assert result['status'] == "success"
            assert result['message'] == "Coordinates standardized and station IDs filled successfully"
            result_data = result['data'][0]
            for station_id_col, coord_cols in station_id_columns_dict.items():
                lat_col, lon_col = coord_cols['coordinate_columns']
                latitude = record[lat_col]
                longitude = record[lon_col]
                station_name_column = coord_cols['station_name_column']
                std_latitude = f"{round(float(latitude), 6):.6f}"
                std_longitude = f"{round(float(longitude), 6):.6f}"
                expected_station_id = f"hashed_station_{std_latitude}_{std_longitude}"
                assert result_data[station_id_col] == expected_station_id
                expected_station_name = record[station_name_column].strip().lower() if pd.notna(record[station_name_column]) else np.nan
                assert result_data[station_name_column] == expected_station_name if pd.notna(expected_station_name) else f"Address for ({std_latitude}, {std_longitude})"
    
    @patch("src.utils.add_station_id")
    @patch("src.utils.get_address")
    def test_non_empty_station_name_and_id(self,
                                           mock_get_address: MagicMock,
                                             mock_add_station_id: MagicMock,
                                             station_id_columns_dict: dict
                                             ) -> None:
        """
        Tests the standardize_coordinates_and_fill_station_ids function when station names and IDs are non-empty.
        It verifies that the function retains existing station names but standardizes the station IDs to follow the hashed256 format.

        The station_name_column is retained as is, while the station_id_column is updated to a hashed value based on the coordinates.
        
        Args:
            None
        
        Returns:
            None
        """
        
        cleaned_df = pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE456"],
            "rideable_type": ["electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", "Station2"],
            "start_station_id": ["Station_id1", "Station_id2"],
            "end_station_name": ["Station3", "Station4"],
            "end_station_id": ["Station_id3", "Station_id4"],
            "start_lat": ["40.7128", "34.0522"],
            "start_lng": ["-74.0060", "-118.2437"],
            "end_lat": ["40.7138", "34.0532"],
            "end_lng": ["-74.0050", "-118.2427"],
            "member_casual": ["member", "casual"]
        })
        
        mock_get_address.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": f"Address for ({lat}, {lng})"
        }
        )
        mock_add_station_id.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": f"hashed_station_{lat}_{lng}"
        }
        )

        cleaned_records = cleaned_df.to_dict(orient="records")
        for record in cleaned_records:
            result = standardize_coordinates_and_fill_station_ids(record, station_id_columns_dict)
            assert result['status'] == "success"
            assert result['message'] == "Coordinates standardized and station IDs filled successfully"
            data = result['data'][0]
            for station_id_col, coord_cols in station_id_columns_dict.items():
                lat_col, lon_col = coord_cols['coordinate_columns']
                latitude = record[lat_col]
                longitude = record[lon_col]
                station_name_column = coord_cols['station_name_column']
                std_latitude = f"{round(float(latitude), 6):.6f}"
                std_longitude = f"{round(float(longitude), 6):.6f}"
                expected_station_name = record[station_name_column]
                assert record[station_name_column] == expected_station_name
                expected_station_id = f"hashed_station_{std_latitude}_{std_longitude}"
                assert data[station_id_col] == expected_station_id
    
    @patch("src.utils.add_station_id")
    @patch("src.utils.get_address")
    def test_empty_end_station_latitude_longitude(self,
                                                   mock_get_address: MagicMock,
                                                   mock_add_station_id: MagicMock,
                                                   station_id_columns_dict: dict) -> None:
        """
        Tests the standardize_coordinates_and_fill_station_ids function when end station latitude or longitude values are empty.
        It verifies that the function leaves station IDs and names unchanged when the end coordinates are missing.

        The start station coordinates cannot be empty from the data validation step. Thus only the end station coordinates are tested here.

        Args:
            - mock_get_address (MagicMock): The patched get_address function.
            - mock_add_station_id (MagicMock): The patched add_station_id function.
            - station_id_columns_dict (dict): A fixture that returns the dictionary mapping station ID columns to their coordinate columns.
        
        Returns:
            None
        """
        cleaned_df = pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE124"],
            "rideable_type": ["electric_bike", "electric_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00"],
            "start_station_name": ["Station1", "Station2"],
            "start_station_id": ["Station_id1", "Station_id2"],
            "end_station_name": [np.nan, np.nan],
            "end_station_id": [np.nan, np.nan],
            "start_lat": ["40.7128", "40.7138"],
            "start_lng": ["-74.0060", "-74.1060"],
            "end_lat": [np.nan, np.nan],  # Empty end latitude
            "end_lng": [np.nan, np.nan],  # Empty end longitude
            "member_casual": ["member", "member"]
        })

        mock_get_address.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": f"Address for ({lat}, {lng})"
        }
        )
        mock_add_station_id.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": f"hashed_station_{lat}_{lng}"
        }
        )
        
        cleaned_records = cleaned_df.to_dict(orient="records")
        for record in cleaned_records:
            result = standardize_coordinates_and_fill_station_ids(record, station_id_columns_dict)
            assert result['status'] == "success"
            assert result['message'] == "Coordinates standardized and station IDs filled successfully"
            data = result['data'][0]
            # Start station ID should be updated
            std_start_latitude = f"{round(float(record['start_lat']), 6):.6f}"
            std_start_longitude = f"{round(float(record['start_lng']), 6):.6f}"
            expected_start_station_id = f"hashed_station_{std_start_latitude}_{std_start_longitude}"
            assert data['start_station_id'] == expected_start_station_id
            # End station ID and name should remain unchanged (NaN)
            assert pd.isna(data['end_station_id']) == True
            assert pd.isna(data['end_station_name']) == True

class TestImputeMissingStationIDs:
    """
    Tests for the impute_missing_station_ids function.
    It uses the standardize_coordinates_and_fill_station_ids function to fill in missing station IDs and names based on coordinates.

    Args:
        None
    
    Returns:
        None
    """ 
    def test_invalid_df_argument_type(self) -> None:
        """
        Tests the impute_missing_station_ids function with an invalid argument type.
        i.e argument is not a dataframe. 
        It should raise an exception when the input is not a pandas DataFrame.
        
        Args:
            None

        Returns:
            None
        """
        invalid_raw_df_types = [
            "invalid_df",
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            123,
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]

        valid_batch_size = 100

        for raw_df_type in invalid_raw_df_types:
                with pytest.raises(Exception) as exc_info:
                    impute_missing_station_ids(raw_df_type, valid_batch_size)  
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while imputing station IDs in bikeshare data"
        assert exc_info.value.args[0]["error"] == "Input df must be a pandas DataFrame"
    

    def test_valid_df_invalid_batch_size(self) -> None:
        """
        Tests the impute_missing_station_ids function with a valid DataFrame but invalid batch sizes.
        It should raise an exception when the batch size is not a positive integer.

        Args:
            None

        Returns:
            None
        """
        valid_raw_df = pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE456"],
            "rideable_type": ["electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", np.nan],
            "start_station_id": ["Station_id1", np.nan],
            "end_station_name": ["Station2", np.nan],
            "end_station_id": ["Station_id2", np.nan],
            "start_lat": ["40.7128", "34.0522"],
            "start_lng": ["-74.0060", "-118.2437"],
            "end_lat": ["40.7138", "34.0532"],
            "end_lng": ["-74.0050", "-118.2427"],
            "member_casual": ["member", "casual"]
        })

        invalid_batch_sizes = [
            -10,
            0,
            3.5,
            "100",
            None,
            [],
            {},
            (1, 2)
        ]

        for batch_size in invalid_batch_sizes:
            with pytest.raises(Exception) as exc_info:
                impute_missing_station_ids(valid_raw_df, batch_size)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while imputing station IDs in bikeshare data"
        assert exc_info.value.args[0]["error"] == "Input batch_size must be a positive integer"
    
    @patch("src.utils.standardize_coordinates_and_fill_station_ids")
    def test_incomplete_dataframe(self,
                            mock_standardize_func: MagicMock) -> None:
        """
        Tests the impute_missing_station_ids function with an invalid dataframe in the DataFrame.
        An invalid dataframe(missing required columns) should raise an exception.

        Args:
            mock_standardize_func (MagicMock): The patched standardize_coordinates_and_fill_station_ids function.

        Returns:
            None
        """

        valid_batch_size = 100
        invalid_dataframe =  pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE456"],
            "rideable_type": ["electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", np.nan],
            "start_station_id": ["Station_id1", np.nan],
            "end_station_name": ["Station2", np.nan],
            "end_station_id": ["Station_id2", np.nan],
            "start_lat": ["40.7128", "34.0522"], 
            "end_lat": ["40.7138", "34.0532"],
            "end_lng": ["-74.0050", "-118.2427"],
            "member_casual": ["member", "casual"]
        })

        missing_column_name = "start_lng"  # 'start_lng' column is missing
        
        mock_standardize_func.side_effect = (
            Exception({
                "status": "error",
                "message": "An error occurred while standardizing coordinates and filling station IDs",
                "error": f"{missing_column_name} column is missing in the record"
            })
        )

        with pytest.raises(Exception) as exc_info:
            impute_missing_station_ids(invalid_dataframe, valid_batch_size)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while imputing station IDs in bikeshare data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "An error occurred while standardizing coordinates and filling station IDs",
            "error": f"{missing_column_name} column is missing in the record"
        }
    

    @patch("src.utils.add_station_id")
    @patch("src.utils.get_address")
    def test_sucessful_imputation(self,
                                    mock_get_address: MagicMock,
                                    mock_add_station_id: MagicMock
                                    ) -> None:

        """
        Tests the impute_missing_station_ids function for successful imputation of missing station IDs and names.
        It verifies that the function correctly processes the DataFrame in batches and fills in missing station IDs and names.
        
        Args:
            mock_standardize_func (MagicMock): The patched standardize_coordinates_and_fill_station_ids function.

        Returns:
            None
        """

        start_lats = ["40.7128", "34.0522", "41.8781"]
        start_lngs = ["-74.0060", "-118.2437", "-87.6298"]  
        end_lats = ["40.7138", np.nan, "41.8791"]
        end_lngs = ["-74.0050", np.nan, "-87.6288"]

        valid_batch_size = 3
        valid_df = {
            "ride_id": ["RIDE123", "RIDE124", "RIDE125"],
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", np.nan, "Station3"],
            "start_station_id": ["Station_id1", np.nan, "Station_id3"],
            "end_station_name": ["Station2", np.nan, "Station4"],
            "end_station_id": ["Station_id2", np.nan, "Station_id4"],
            "start_lat": [f"{start_lats[0]}", f"{start_lats[1]}", f"{start_lats[2]}"],
            "start_lng": [f"{start_lngs[0]}", f"{start_lngs[1]}", f"{start_lngs[2]}"],
            "end_lat": [f"{end_lats[0]}", f"{end_lats[1]}", f"{end_lats[2]}"],
            "end_lng": [f"{end_lngs[0]}", f"{end_lngs[1]}", f"{end_lngs[2]}"],
            "member_casual": ["member", "member", "casual"]
        }
        
        mock_get_address.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": f"station_name_for_{lat}_{lng}"
        } 

        )

        mock_add_station_id.side_effect = (
        lambda lat, lng: {
            "status": "success",
            "message": "Station ID generated successfully",
            "data": f"hashed_station_{lat}_{lng}"
        }
        )

        valid_df = pd.DataFrame(valid_df)
        result = impute_missing_station_ids(valid_df, valid_batch_size)
        assert result['status'] == "success"
        assert result['message'] == f"Successfully processed bikeshare data with imputed station IDs."
        assert result['data'].shape == valid_df.shape
        std_start_lats = [f"{round(float(lat), 6):.6f}" for lat in start_lats]
        std_start_lngs = [f"{round(float(lng), 6):.6f}" for lng in start_lngs]
        std_end_lats = [f"{round(float(lat), 6):.6f}" if pd.notna(lat) else lat for lat in end_lats]
        std_end_lngs = [f"{round(float(lng), 6):.6f}" if pd.notna(lng) else lng for lng in end_lngs]

        expected_processed_df = pd.DataFrame({
            "ride_id": ["RIDE123", "RIDE124", "RIDE125"],
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": ["2022-12-01 08:00:00", "2022-12-01 08:05:00", "2022-12-01 09:00:00"],
            "ended_at": ["2022-12-01 08:30:00", "2022-12-01 08:35:00", "2022-12-01 09:30:00"],
            "start_station_name": ["Station1", f"station_name_for_{std_start_lats[1]}_{std_start_lngs[1]}", "Station3"],
            "start_station_id": [f"hashed_station_{std_start_lats[0]}_{std_start_lngs[0]}", f"hashed_station_{std_start_lats[1]}_{std_start_lngs[1]}", f"hashed_station_{std_start_lats[2]}_{std_start_lngs[2]}"],
            "end_station_name": [f"Station2", np.nan, "Station4"],
            "end_station_id": [f"hashed_station_{std_end_lats[0]}_{std_end_lngs[0]}", np.nan, f"hashed_station_{std_end_lats[2]}_{std_end_lngs[2]}"],
            "start_lat": [std_start_lats[0], std_start_lats[1], std_start_lats[2]],
            "start_lng": [std_start_lngs[0], std_start_lngs[1], std_start_lngs[2]],
            "end_lat": [std_end_lats[0], std_end_lats[1], std_end_lats[2]],
            "end_lng": [std_end_lngs[0], std_end_lngs[1], std_end_lngs[2]],
            "member_casual": ["member", "member", "casual"]
        })
        expected_processed_df[[
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_station_name", "start_station_id", "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]]

        end_latss = result['data']['end_lat'].tolist()
        end_latss = [(lat, type(lat)) for lat in end_latss]

        station_id_columns = ['start_station_id', 'end_station_id', 'start_station_name', 'end_station_name']

        for col in station_id_columns:
            expected_processed_df[col] = expected_processed_df[col].apply(lambda x: x.strip().lower() if pd.notna(x) else x)

        result_normalized = result['data'].reset_index(drop=True).fillna("None")
        expected_normalized = expected_processed_df.reset_index(drop=True).fillna("None")
        pd.testing.assert_frame_equal(result_normalized,
                                               expected_normalized)

class TestValidateProcessedData:
    """
    Tests for the validate_processed_data function.
    """ 
    def test_invalid_argument_type(self) -> None:
        """
        Tests the validate_processed_data function with an invalid argument type.
        Here, the "s3_config" argument is not a dictionary.

        Args:
            None

        Returns:
            None
        """
        invalid_s3_config_types = [ 
            "invalid_df",
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            123,
            None,
            [],
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_processed_s3_key = "processed_data/test_file.parquet"

        for s3_config in invalid_s3_config_types:
            with pytest.raises(Exception) as exc_info:
                validate_processed_data(s3_config, valid_processed_s3_key) #s3_config must be a dictionary
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == "Input s3_config must be a dictionary"
    
    def test_invalid_processed_s3_key_type(self) -> None:
        """
        Tests the validate_processed_data function with an invalid processed_s3_key type.
        Here, the "processed_s3_key" argument is not a string.
        
        Args:
            None
        
        Returns:
            None
        """

        invalid_processed_s3_key_types = [ 
            "invalid_key",
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            123,
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_s3_config = {
            "bucket_name": "test-bucket",
            "aws_access_key_id": "TESTACCESSKEY",
            "aws_secret_access_key": "TESTSECRETKEY",
            "region_name": "us-east-1"
        }

        for processed_s3_key in invalid_processed_s3_key_types:
            with pytest.raises(Exception) as exc_info:
                validate_processed_data(valid_s3_config, processed_s3_key) #processed_s3_key must be a string
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == "Input processed_data_key must be a string"
    

    # def test_load_transformed_bucket(self, 
    #                         moto_boto3_client: Session,
    #                         aws_credentials: dict) -> None:
    #     """
    #     This function sets up the mock raw S3 bucket and uploads a sample CSV file to it.

    #     Args:
    #         moto_boto3_client: Fixture providing a mocked boto3 client.
    #         aws_credentials: Fixture providing mocked AWS credentials.
        
    #     Returns:
    #         None
    #     """  

    #     bucket = aws_credentials['transformed_bucket']
    #     s3_key = aws_credentials['transformed_s3_key']
    #     moto_boto3_client.create_bucket(Bucket=bucket)
        
    #     df = pd.DataFrame({
    #         "ride_id": ["ride123", "ride124", "ride125"],
    #         "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
    #         "started_at": [
    #             "2022-12-01 08:00:00",
    #             "2022-12-01 08:05:00",
    #             "2022-12-01 09:00:00",
    #         ],
    #         "ended_at": [
    #             "2022-12-01 08:30:00",
    #             "2022-12-01 08:35:00",
    #             "2022-12-01 09:30:00",
    #         ],
    #         "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
    #         "start_station_id": ["st001", "st002", "st003"],
    #         "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
    #         "end_station_id": ["en001", "en002", "en003"],
    #         "start_lat": [40.7128, 34.0522, 41.8781],
    #         "start_lng": [-74.0060, -118.2437, -87.6298],
    #         "end_lat": [40.7138, 35.0522, 41.8791],
    #         "end_lng": [-74.0050, -118.2437, -87.6288],
    #         "member_casual": ["member", "member", "casual"],
    #     })

    #     buffer = BytesIO()
    #     df.to_parquet(buffer, index=False)
    #     buffer.seek(0)

    #     moto_boto3_client.put_object(
    #         Bucket=bucket,
    #         Key=s3_key,
    #         Body=buffer.read()
    #     )
        
    # def test_validation_success(self, 
    #                             aws_credentials: dict,
    #                             mock_boto3_client: MagicMock
    #                             ) -> None:
    #     """
    #     Tests the validate_processed_data function for successful validation of processed data in S3.
    #     It verifies that the function correctly validates the existence and schema of the processed data file in S3.

    #     Args:
    #         s3_config (dict): A fixture that returns a valid S3 configuration dictionary.
    #         processed_s3_key (str): A fixture that returns a valid S3 key for the processed data file.

    #     Returns:
    #         None
    #     """
    #     print("aws_credentials", aws_credentials)
    #     s3_config = {
    #         "transformed_bucket": aws_credentials['transformed_bucket'],
    #         "access_key": aws_credentials["aws_access_key"],
    #         "secret_key": aws_credentials["aws_secret_key"],
    #         "region_name": "us-east-1"
    #     }
    #     processed_s3_key = aws_credentials['transformed_s3_key']

    #     with patch('boto3.client', mock_boto3_client) as mock_client:
    #         result = validate_processed_data(s3_config, processed_s3_key)

    #     assert result['status'] == "success"


    def test_missing_required_columns(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when required columns are missing in the processed data file in S3.
        It verifies that the function raises an exception when the processed data file does not contain all required columns.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.

        Returns:
            None
        """        
        missing_columns = ["start_station_id", "end_station_name"]

        valid_df = pd.DataFrame({
            "ride_id": ["ride123", "ride124", "ride125"],
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })
        invalid_df = valid_df.drop(columns=missing_columns)

        buffer = BytesIO()
        invalid_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(invalid_df)
        
        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": f"columns: {sorted(missing_columns)} are missing"
        }
    
    def test_non_null_columns(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when non-nullable columns contain null values in the processed data file in S3.
        It verifies that the function raises an exception when non-nullable columns have null values.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.

        Returns:
            None
        """

        df = pd.DataFrame({
            "ride_id": ["ride123", "ride124", np.nan], # Introducing null value in 'ride_id'
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, np.nan], # Introducing null value in 'start_lat'
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)
        
        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "At least one non_null column contains nulls"
        }

    def test_unique_ride_id_validation(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when the 'ride_id' column contains duplicate values in the processed data file in S3.
        It verifies that the function raises an exception when 'ride_id' values are not unique.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
        Returns:
            None
        """

        df = pd.DataFrame({
            "ride_id": ["ride123", "ride124", "ride123"], # Duplicate
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)

        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "ride_id column contains duplicate values"
        }
    
    def test_datetime_columns_validation(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when datetime columns contain invalid datetime strings in the processed data file in S3.
        It verifies that the function raises an exception when datetime columns have invalid formats.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
        
        Returns:
            None
        """
        df = pd.DataFrame({
            "ride_id": ["ride123", "ride124", "ride125"],
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "invalid_datetime",  # Invalid datetime string
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "another_invalid_datetime",  # Invalid datetime string
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)

        invalid_datetime_columns = ["started_at", "ended_at"]

        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": f"{invalid_datetime_columns} contains invalid datetime format" 
        }
    
    def test_str_columns_validation(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when string columns contain non-string values in the processed data file in S3.
        It verifies that the function raises an exception when string columns have non-string types.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
        
        Returns:
            None
        """
        df = pd.DataFrame({
            "ride_id": [123, 124, 125],  # Non-string value
            "rideable_type": [10.1, 10.2, 10.3], # Non-string value
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })
        failed_str_columns = ["ride_id", "rideable_type"]
        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)

        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": f"{failed_str_columns} contains non-string values"
        }
    
    def test_ride_id_length_validation(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when 'ride_id' column contains values with incorrect length in the processed data file in S3.
        It verifies that the function raises an exception when 'ride_id' values do not have the expected length.

        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
        
        Returns:
            None
        """
        df = pd.DataFrame({
            "ride_id": ["ride1", "ride23456", "ride123"],  # Incorrect lengths
            "rideable_type": ["electric_bike", "electric_bike", "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)
        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "ride_id column contains values with incorrect length"
        }
    

    def test_start_station_id_length_validation(self,
                            create_transformed_bucket: Dict,
                            upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
                            ) -> None:
        """
        Tests the validate_processed_data function when 'start_station_id' column contains values with incorrect length in the processed data file in S3.
        It verifies that the function raises an exception when 'start_station_id' values do not have the expected length.

        Args:
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
            upload_transformed_data_to_s3: Fixture that uploads the provided DataFrame to the transformed bucket. 

        Returns:
            None
        """
        df = pd.DataFrame({
            "ride_id": ["1111111111111111",
                         "2222222222222222",
                           "3333333333333333"],  # correct lengths(16)
            "rideable_type": ["electric_bike", 
                              "electric_bike",
                                "classic_bike"],
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00",
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"],
            "start_station_id": ["st001", "st002", "st003"], #incorrect lengths (!= 16)
            "end_station_name": ["end_station_1", "end_station_2", "end_station_3"],
            "end_station_id": ["en001", "en002", "en003"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, 41.8791],
            "end_lng": [-74.0050, -118.2437, -87.6288],
            "member_casual": ["member", "member", "casual"],
        })

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)
        with pytest.raises(Exception) as exc_info:
            validate_processed_data(s3_config, processed_s3_key)
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while validating processed data"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "start_station_id column contains values with incorrect length"
        }


    def test_validation_success(
        self, 
        create_transformed_bucket: Dict,
        upload_transformed_data_to_s3: Callable[[pd.DataFrame], None]
        ) -> None:
        
        """
        Tests the validate_processed_data function for successful validation of processed data in S3.

        Args: 
            create_transformed_bucket: Fixture that sets up the mock transformed S3 bucket.
            upload_transformed_data_to_s3: Fixture that uploads the provided DataFrame to the transformed bucket.
        Returns:
            None
        """

        df = pd.DataFrame({
            "ride_id": ["1111111111111111",
                         "2222222222222222",
                           "3333333333333333"],  # correct lengths(16) and type
            "rideable_type": ["electric_bike", 
                              "electric_bike",
                                "classic_bike"], #correct type
            "started_at": [
                "2022-12-01 08:00:00",
                "2022-12-01 08:05:00",
                "2022-12-01 09:00:00",   #correct datetime format
            ],
            "ended_at": [
                "2022-12-01 08:30:00",
                "2022-12-01 08:35:00", #correct datetime format
                "2022-12-01 09:30:00",
            ],
            "start_station_name": ["start_station_1", "start_station_2", "start_station_3"], #correct type
            "end_station_name": ["end_station_1", "end_station_2", np.nan], #correct type
            "end_station_id": ["hashed_end_s_1", "hashed_end_s_2", np.nan], #correct type
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 35.0522, np.nan],
            "end_lng": [-74.0050, -118.2437, np.nan],
            "member_casual": ["member", "member", "casual"],
        })
        
        def random_64_string() -> str:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=64))

        df['start_station_id'] = [random_64_string() for _ in range(len(df))]

        s3_config = create_transformed_bucket
        processed_s3_key = s3_config['transformed_s3_key']
        upload_transformed_data_to_s3(df)
        result = validate_processed_data(s3_config, processed_s3_key)
        
        assert result['status'] == "success"
        assert result['message'] == "Processed data validation completed successfully"
        assert result['data_shape'] == df.shape

# class TestSendSlackAlert:
#     """
#     Tests for the send_slack_alert function.
    
#     """

#     def test_invalid_message_argument_type(
#         self
#     ) -> None:
#         """
#         Tests the send_slack_alert function with an invalid message argument type.
#         Here, the "message" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_message_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         valid_channel = "#test-channel"

#         for message in invalid_message_types:
#             with pytest.raises(Exception) as exc_info:
#                 send_slack_alert(message=message, 
#                                 channel=valid_channel,
#                                 oauth_token="xoxb-test-token",
#                                 slack_bot="test-bot") 
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
#         assert exc_info.value.args[0]['error'] == "message argument must be a string"

#     def test_invalid_channel_argument_type(
#         self
#     ) -> None:
#         """
#         Tests the send_slack_alert function with an invalid channel argument type.
#         Here, the "channel" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_channel_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         valid_message = "Test Slack alert message"

#         for channel in invalid_channel_types:
#             with pytest.raises(Exception) as exc_info:
#                 send_slack_alert(message=valid_message, 
#                                 channel=channel,
#                                 oauth_token="xoxb-test-token",
#                                 slack_bot="test-bot") 
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
#         assert exc_info.value.args[0]['error'] == "channel argument must be a string"
    

#     def test_invalid_oauth_token_argument_type(
#         self
#     ) -> None:
        
#         """
#         Tests the send_slack_alert function with an invalid oauth_token argument type.
#         Here, the "oauth_token" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_oauth_token_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         valid_message = "Test Slack alert message"
#         valid_channel = "#test-channel"

#         for oauth_token in invalid_oauth_token_types:
#             with pytest.raises(Exception) as exc_info:
#                 send_slack_alert(message=valid_message, 
#                                 channel=valid_channel,
#                                 oauth_token=oauth_token,
#                                 slack_bot="test-bot") 
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
#         assert exc_info.value.args[0]['error'] == "oauth_token argument must be a string"

#     def test_invalid_slack_bot_argument_type(
#         self
#     ) -> None:
#         """
#         Tests the send_slack_alert function with an invalid slack_bot argument type.
#         Here, the "slack_bot" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_slack_bot_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         valid_message = "Test Slack alert message"
#         valid_channel = "#test-channel"
#         valid_oauth_token = "xoxb-test-token"

#         for slack_bot in invalid_slack_bot_types:
#             with pytest.raises(Exception) as exc_info:
#                 send_slack_alert(message=valid_message, 
#                                 channel=valid_channel,
#                                 oauth_token=valid_oauth_token,
#                                 slack_bot=slack_bot) 
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
#         assert exc_info.value.args[0]['error'] == "slack_bot argument must be a string"


#     @patch('src.utils.WebClient')
#     def test_send_slack_alert_failure(self, 
#                                       mock_factory: MagicMock) -> None:
#         """
#         Tests the send_slack_alert function for failed Slack alert sending.
#         It verifies that the function correctly handles a failure response from Slack API.

#         Args:
#             mock_factory: Mocked WebClient factory.

#         Returns:
#             None
#         """
#         mock_session = MagicMock()
#         mock_session.chat_postMessage.side_effect = Exception("The request to the slack API failed.")
#         mock_factory.return_value = mock_session

#         message = "Test Slack alert message"
#         channel = "#test-channel"
#         with pytest.raises(Exception) as exc_info:
#             send_slack_alert(message=message, 
#                             channel="#test-channel",
#                             oauth_token="xoxb-test-token",
#                             slack_bot="test-bot")
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == f"An error occurred while sending alert to Slack"
#         assert exc_info.value.args[0]['error'] == 'The request to the slack API failed.'
      

#     @patch('src.utils.WebClient')
#     def test_send_slack_alert_success(self, mock_factory: MagicMock) -> None:
#         """
#         Tests the send_slack_alert function for successful Slack alert sending.
#         It verifies that the function correctly sends a Slack alert and returns a success response.

#         Args:
#             mock_factory: Mocked WebClient factory.

#         Returns:
#             None
#         """
#         mock_session = MagicMock()
#         mock_response = MagicMock()
#         mock_response.ok.return_value = True
#         mock_session.chat_postMessage.return_value = mock_response
#         mock_factory.return_value = mock_session

#         message = "Test Slack alert message"
#         channel = "#test-channel"
#         result = send_slack_alert(message=message, 
#                                   channel="#test-channel",
#                                   oauth_token="xoxb-test-token",
#                                   slack_bot="test-bot")

#         assert result['status'] == "success"
#         assert result['message'] == f"Alert sent to Slack: {channel} successfully"
    

# class TestProcessStream:

#     def test_invalid_data_path_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid data_path argument type.
#         Here, the "data_path" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_data_path_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         channel_id = "#test-channel"
#         oauth_token = "xoxb-test-token"
#         slack_bot_name = "test-bot"
#         delay_seconds = 2
#         chunk_size = 10

        
#         for data_path in invalid_data_path_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                  flagged_events_log_path=flagged_events_log_path,
#                                  channel_id=channel_id,
#                                  oauth_token=oauth_token,
#                                  slack_bot_name=slack_bot_name,
#                                  delay_seconds=delay_seconds,
#                                  chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "data_path argument must be a string"

#     def test_invalid_flagged_events_log_path_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid flagged_events_log_path argument type.
#         Here, the "flagged_events_log_path" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_flagged_events_log_path_types = [ 
#             "123",
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         channel_id = "#test-channel"
#         oauth_token = "xoxb-test-token"
#         slack_bot_name = "test-bot"
#         delay_seconds = 2
#         chunk_size = 10

#         for flagged_events_log_path in invalid_flagged_events_log_path_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path="valid/path/to/data.csv",
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "flagged_events_log_path argument must be a string"
    
#     def test_invalid_channel_id_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid channel_id argument type.
#         Here, the "channel_id" argument must be a string to pass validation.
#         Args:
#             None
#         Returns:
#             None
#         """

#         invalid_channel_id_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         oauth_token = "xoxb-test-token"
#         slack_bot_name = "test-bot"
#         delay_seconds = 2
#         chunk_size = 10

#         for channel_id in invalid_channel_id_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "channel_id argument must be a string"
    
#     def test_invalid_oauth_token_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid oauth_token argument type.

#         Here, the "oauth_token" argument must be a string to pass validation.
#         Args:
#             None
#         Returns:
#             None
#         """
#         invalid_oauth_token_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         channel_id = "#test-channel"
#         slack_bot_name = "test-bot"
#         delay_seconds = 2
#         chunk_size = 10

#         for oauth_token in invalid_oauth_token_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "oauth_token argument must be a string"
    
#     def test_invalid_slack_bot_name_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid slack_bot_name argument type.
#         Here, the "slack_bot_name" argument must be a string to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_slack_bot_name_types = [ 
#             123,
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         channel_id = "#test-channel"
#         oauth_token = "xoxb-test-token"
#         delay_seconds = 2
#         chunk_size = 10

#         for slack_bot_name in invalid_slack_bot_name_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "slack_bot_name argument must be a string"
    
#     def test_invalid_delay_seconds_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid delay_seconds argument type.
#         Here, the "delay_seconds" argument must be an integer to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_delay_seconds_types = [ 
#             "123",
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         channel_id = "#test-channel"
#         oauth_token = "xoxb-test-token"
#         slack_bot_name = "test-bot"
#         chunk_size = 10

#         for delay_seconds in invalid_delay_seconds_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "delay_seconds argument must be an integer"

#     def test_invalid_chunk_size_argument(self) -> None:
#         """
#         Tests the process_stream function with an invalid chunk_size argument type.

#         Here, the "chunk_size" argument must be an integer to pass validation.

#         Args:
#             None

#         Returns:
#             None
#         """
#         invalid_chunk_size_types = [ 
#             "123",
#             pd.Series([1, 2, 3]),
#             np.array([[1, 2], [3, 4]]),
#             None,
#             [],
#             {},
#             True,
#             False,
#             12.34,
#             (1, 2, 3) 
#         ]
#         data_path = "valid/path/to/data.csv"
#         flagged_events_log_path = "valid/path/to/flagged_events.log"
#         channel_id = "#test-channel"
#         oauth_token = "xoxb-test-token"
#         slack_bot_name = "test-bot"
#         delay_seconds = 2

#         for chunk_size in invalid_chunk_size_types:
#             with pytest.raises(Exception) as exc_info:
#                 process_stream(data_path=data_path,
#                                flagged_events_log_path=flagged_events_log_path,
#                                channel_id=channel_id,
#                                oauth_token=oauth_token,
#                                slack_bot_name=slack_bot_name,
#                                delay_seconds=delay_seconds,
#                                chunk_size=chunk_size)
        
#         assert exc_info.type is Exception
#         assert exc_info.value.args[0]['status'] == "error"
#         assert exc_info.value.args[0]['message'] == "An error occurred while processing event stream from raw data"
#         assert exc_info.value.args[0]['error'] == "chunk_size argument must be an integer"


    
