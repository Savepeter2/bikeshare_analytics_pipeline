import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (gen_hash_key_station_id, extract_source_data,
                        validate_raw_data, requests_session_with_retries, get_address,
                        add_station_id, clean_raw_data, standardize_coordinates_and_fill_station_ids,
                        impute_missing_station_ids, validate_processed_data, send_slack_alert,
                        list_partitions, write_flagged_event_to_snowflake
                        )

from typing import Dict, Callable, Generator
from unittest.mock import patch, MagicMock
from botocore.client import BaseClient
from botocore.exceptions import ClientError, NoCredentialsError
from boto3.session import Session
from pytest import MonkeyPatch
from requests.adapters import HTTPAdapter, Retry
from requests.sessions import Session
from requests.exceptions import ConnectionError
from sqlalchemy.exc import OperationalError, InterfaceError, ProgrammingError
import random
import string
import ast
from moto import mock_aws
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from src.utils import yield_event_stream





@pytest.fixture(scope = "function")
def mimick_s3_page() -> Callable:
    def make_s3_page(keys: list[str]) -> dict:
        """Build a fake paginator page from a list of S3 object keys."""
        return {"Contents": [{"Key": k} for k in keys]}

    return make_s3_page

@pytest.fixture(scope="function")
def make_empty_page() -> dict:
    """Simulate an empty S3 prefix (no Contents key)."""
    return {}


@pytest.fixture(scope="class")
def aws_credentials() -> Dict[str, str]:
    """
    These are FAKE credentials used only for mocking.
    They will never connect to real AWS.

    Args:
        None

    Returns:
        dict: A dictionary containing fake AWS credentials and S3 bucket/key information for testing.
    """
    return {
        'aws_access_key': 'valid_access_key',
        'aws_secret_key': 'valid_secret_key',
        'raw_bucket': "valid_raw_bucket",
        'raw_s3_key': "raw/raw_data.parquet",
        'transformed_bucket': "transformed-bucket",
        'transformed_s3_key': "transformed/staging_data.parquet",
        'source_bucket': "valid_source_bucket",
        'source_s3_key': "valid_source_s3_key",
        'raw_bucket_folder':  "valid_raw_bucket_folder",
        'raw_bucket_weekly_dump_prefix': 'valid_raw_weekly_dump_prefix',
        'raw_bucket_metadata_prefix': 'valid_raw_metadata_prefix',
        'transformed_bucket_metadata_prefix': 'valid_transformed_metadata_prefix'
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

@pytest.fixture(scope="class")
def valid_snowflake_config():
    return {
        "snowflake_account": "test_account",
        "snowflake_username": "test_user",
        "snowflake_password": "test_password",
        "snowflake_warehouse": "test_warehouse",
        "snowflake_database": "test_database",
        "snowflake_schema": "test_schema",
        "snowflake_role": "test_role",
        "stage_name": "test_stage",
    }

@pytest.fixture(autouse=True, scope='class')
def valid_event() -> None:
    return {
        "id":                 "evt-001",
        "ride_id":            "abc123",
        "flag_type":          "midnight_casual_rider",
        "rideable_type":      "electric_bike",
        "started_at":         "2022-11-28 00:05:00",
        "ended_at":           "2022-11-28 01:05:00",
        "start_station_name": "Main St",
        "start_station_id":   "ST001",
        "end_station_name":   "Park Ave",
        "end_station_id":     "ST002",
        "start_lat":          41.8781,
        "start_lng":          -87.6298,
        "end_lat":            41.8820,
        "end_lng":            -87.6230,
        "member_casual":      "casual",
    }



def _setup_mock_alerts_log(mock_alerts_log: MagicMock) -> None:
    """
    AlertsLog is a SQLAlchemy model. We need __table_args__ and __tablename__
    to exist so the INSERT string is rendered without AttributeError.
    """
    mock_alerts_log.__table_args__ = {"schema": "ALERTS"}
    mock_alerts_log.__tablename__  = "alerts_log"


def _setup_mock_session(mock_session: MagicMock):
    """
    create_session is used as a context manager (`with create_session(...) as db`).
    Returns (mock_db, mock_ctx) so individual tests can configure side effects.
    """
    mock_db  = MagicMock()
    mock_ctx = mock_session.return_value.__enter__.return_value = mock_db
    mock_session.return_value.__exit__.return_value = False  # don't suppress exceptions
    return mock_db, mock_ctx         

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
        invalid_access_key_types = [123, None, [], {}, True, False, 12.34]
        valid_arguments_types = ("valid_secret_key", "source-bucket", 
                                 "source-key", "batch-year", "batch-week", 100000, {})

        for inv_key in invalid_access_key_types:
            args = (inv_key,) + valid_arguments_types
            aws_access_key, aws_secret_key, source_bucket, \
                source_s3_key, batch_year, batch_week, chunk_size, raw_data_schema = args

            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
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

        invalid_secret_key_type = [123, None, [], {}, True, False, 12.34]
        other_valid_arguments_type = ("valid_access_key", "test-bucket", "test-key", "batch_year", "batch_week", 100000, {})
        
        for inv_secr in invalid_secret_key_type:
            args = (other_valid_arguments_type[0], inv_secr) + other_valid_arguments_type[1:]

            aws_access_key, aws_secret_key, source_bucket, source_s3_key, \
                batch_year, batch_week, chunk_size, raw_data_schema = args
            
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
    def test_extract_data_invalid_source_bucket_type(self):
        """
        Tests the extract_source_data function with invalid source bucket types.

        Returns:
            None
        """
        invalid_source_bucket_type = [123, None, [], {}, True, False, 12.34]
        other_valid_arguments_type = ("valid_access_key", "valid_secret_key", 
                                      "test-source-key", "batch_year",
                                        "batch_week", 100000, {})
        
        for inv_src_bucket_type in invalid_source_bucket_type:
            args = (other_valid_arguments_type[:2] + (inv_src_bucket_type,) + other_valid_arguments_type[2:])
            aws_access_key, aws_secret_key, source_bucket,\
                  source_s3_key, batch_year, batch_week, chunk_size, raw_data_schema = args
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
    def test_extract_data_invalid_s3_key_type(self):
        """
        Tests the extract_source_data function with invalid source s3 key types.

        Returns:
            None
        """
        invalid_source_s3_key_type = [123, None, [], {}, True, False, 12.34]
        other_valid_arguments_type = ("valid_access_key", "valid_secret_key", 
                                      "test-source-key", "batch_year",
                                        "batch_week", 100000, {})
        
        for inv_key_type in invalid_source_s3_key_type:
            args = (other_valid_arguments_type[:3] + (inv_key_type,) + other_valid_arguments_type[3:])
            aws_access_key, aws_secret_key, source_bucket,\
                  source_s3_key, batch_year, batch_week, chunk_size, raw_data_schema = args
            
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "AWS access key, secret key, source bucket, and source s3 key must be strings"
            }
    
    def test_extract_data_invalid_batch_year_type(self):
        """
        Tests the extract_source_data function with invalid batch year types.

        Returns:
            None
        """
        invalid_batch_year_type = [123, None, [],
                                       {}, True, False, 12.34]
        
        other_valid_arguments_type = ("valid_access_key", "valid_secret_key", 
                                      "test-source-key", "batch_year",
                                        "batch_week", 100000, {})
        
        for inv_key_type in invalid_batch_year_type:
            args = (other_valid_arguments_type[:4] + (inv_key_type,) + other_valid_arguments_type[4:])
            aws_access_key, aws_secret_key, source_bucket,\
                  source_s3_key, batch_year, batch_week, chunk_size, raw_data_schema = args
            
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "Batch year must be a string"
            }
    
    def test_extract_data_invalid_batch_week_type(self):
        """
        Tests the extract_source_data function with invalid batch week types.
        
        Returns:
            None

        """
        invalid_batch_week_type = [123, None, [],
                                       {}, True, False, 12.34]
        
        other_valid_arguments_type = ("valid_access_key", "valid_secret_key",
                                      "test-source-bucket", 
                                      "test-source-key", "batch_year", 100000, {})
        
        for inv_key_type in invalid_batch_week_type:
            args = (other_valid_arguments_type[:5] + (inv_key_type,) + other_valid_arguments_type[5:])
            aws_access_key, aws_secret_key, source_bucket,\
                  source_s3_key, batch_year, batch_week, chunk_size, raw_data_schema = args
        
            with pytest.raises(Exception) as exc_info:
                extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=batch_year,
                    batch_week=batch_week,
                    chunk_size=chunk_size,
                    raw_data_schema=raw_data_schema
                )
            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while extracting data from source s3",
                "error": "Batch week must be a string"
            }

    def test_load_source_bucket(self, 
                    moto_boto3_client: Session,
                    aws_credentials: dict) -> None:
        """
        This function sets up the mock source S3 bucket and uploads a sample CSV file to it.
        Args:
            moto_boto3_client: Fixture providing a mocked boto3 client.
            aws_credentials: Fixture providing mocked AWS credentials.
        
        Returns:
            None
        """  
    
        bucket = aws_credentials['source_bucket']
        s3_key = aws_credentials['source_s3_key']
        moto_boto3_client.create_bucket(Bucket=bucket)
        moto_boto3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body = b"""ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual
ride_id1,electric_bike,2024-12-01 08:00,2024-12-01 08:15,Station A,start_station_id1,Station B,end_station_id1,40.7128,-74.0060,40.7150,-74.0020,member
ride_id2,classic_bike,2024-12-01 09:00,2024-12-01 09:25,Station C,start_station_id2,Station D,end_station_id2,40.7306,-73.9352,40.7320,-73.9300,casual
ride_id3,docked_bike,2024-12-01 10:00,2024-12-01 10:45,Station E,start_station_id3,Station F,end_station_id3,40.7580,-73.9855,40.7600,-73.9800,member
                """
    #sample rides here are all in december 1st, 2024 week 48
    )
        
    def test_extract_data_unauthorized_credentials(self,
                                        mock_boto3_client: Session,
                                        aws_credentials: dict) -> None:
        
        """
        Tests the extract_source_data function with unauthorized AWS credentials.
        It uses moto to mock S3 interactions.

        Args:
            mock_boto3_client: Fixture providing a mocked boto3 client.
            aws_credentials: Fixture providing mocked AWS credentials.

        Returns:
            None
        """
        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']

        invalid_aws_credentials = [
            ("INVALIDACCESSKEY", aws_credentials['aws_secret_key']),
            (aws_credentials['aws_access_key'], "INVALIDSECRETKEY"),
            ("INVALIDACCESSKEY", "INVALIDSECRETKEY"),
        ]
        valid_batch_year = "2024"
        valid_batch_week = "48"

        for aws_access_key, aws_secret_key in invalid_aws_credentials:
            with patch('boto3.client', side_effect=mock_boto3_client):
                with pytest.raises(Exception) as exc_info:
                    extract_source_data(
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key, 
                        source_bucket=source_bucket,
                        source_s3_key=source_s3_key,
                        batch_year=valid_batch_year,
                        batch_week=valid_batch_week,
                        chunk_size=100000,
                        raw_data_schema={}
                    )
                assert exc_info.type is Exception
                assert exc_info.value.args[0]["status"] == "error"
                assert "The security token included in the request is invalid" in exc_info.value.args[0]["error"]
    
    @patch('src.utils.boto3.client')
    def test_extract_data_exception_from_boto3_client(self,
                                        mock_boto3_client: Session,
                                        aws_credentials: dict) -> None:
        """
        Tests the extract_source_data function when the boto3 client throws an exception.
        It uses the mock_boto3_client fixture to mock S3 interactions and simulates an exception being thrown by the boto3 client.

        Args:
            mock_boto3_client: Fixture providing a mocked boto3 client that raises an exception.
            aws_credentials: Fixture providing mocked AWS credentials.

        Returns:
            None
        """
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.side_effect = Exception("An error occurred while running get_object on boto3 client")
        
        aws_access_key = aws_credentials['aws_access_key']
        aws_secret_key = aws_credentials['aws_secret_key']
        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']
        valid_batch_year = "2024"
        valid_batch_week = "48"


        with pytest.raises(Exception) as exc_info:
            
            extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key, 
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=valid_batch_year,
                    batch_week=valid_batch_week,
                    chunk_size=100000,
                    raw_data_schema={}
                )
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while extracting data from source s3"
        assert exc_info.value.args[0]["error"] == "An error occurred while running get_object on boto3 client"
    
    @patch('src.utils.boto3.client')
    def test_extract_data_general_exception(self,
                                        mock_boto3_client: Session,
                                        aws_credentials: dict) -> None:
        """
        Tests the extract_source_data function when a general exception is thrown.
        It uses the mock_boto3_client fixture to mock S3 interactions and simulates a general exception being thrown.

        Args:
            mock_boto3_client: Fixture providing a mocked boto3 client that raises an exception.
            aws_credentials: Fixture providing mocked AWS credentials.

        Returns:
            None
        """
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_object.side_effect = Exception("A exception occurred when extracting source data")

        aws_access_key = aws_credentials['aws_access_key']
        aws_secret_key = aws_credentials['aws_secret_key']
        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']
        valid_batch_year = "2024"
        valid_batch_week = "48"
        with pytest.raises(Exception) as exc_info:
            extract_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key, 
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_year=valid_batch_year,
                    batch_week=valid_batch_week,
                    chunk_size=100000,
                    raw_data_schema={}
                )
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while extracting data from source s3"
        assert exc_info.value.args[0]["error"] == "A exception occurred when extracting source data"
    
    def test_extract_data_no_data_for_batch(self,
                                        mock_boto3_client: Session,
                                        aws_credentials: dict) -> None:
        """
        Tests the extract_source_data function when no data is found for the specified batch year and week.
        It uses moto to mock S3 interactions.

        Args:
            mock_boto3_client: Fixture providing a mocked boto3 client.
            aws_credentials: Fixture providing mocked AWS credentials.

        Returns:
            None
        """

        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']
        aws_access_key = aws_credentials['aws_access_key']
        aws_secret_key = aws_credentials['aws_secret_key']
        batch_year_with_no_data = "2023"
        batch_week_with_no_data = "01"

        with patch('boto3.client', side_effect=mock_boto3_client):
            result = extract_source_data(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key, 
                source_bucket=source_bucket,
                source_s3_key=source_s3_key,
                batch_year=batch_year_with_no_data,
                batch_week=batch_week_with_no_data,
                chunk_size=100000,
                raw_data_schema={}
            )

            assert result["status"] == "success"
            assert result["message"] == f"No data found for {batch_year_with_no_data}-W{batch_week_with_no_data} in source s3 bucket: {source_bucket}"
            assert result['data'].empty == True

    def test_extract_data_success(self, 
                                      mock_boto3_client: Session,
                                      aws_credentials: dict) -> None:
        """
        Tests the extract_source_data function for a successful batch week extraction where data is present.
        It uses moto to mock S3 interactions.
        
        Args:
            mock_boto3_client: Fixture providing a mocked boto3 client.
        
        Returns:
            None
        """
        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']
        valid_batch_year = "2024"
        valid_batch_week = "48"
        chunk_size = 100000
        
        with patch('boto3.client', side_effect=mock_boto3_client):
            result = extract_source_data(
                aws_access_key=aws_credentials['aws_access_key'],
                aws_secret_key=aws_credentials['aws_secret_key'], 
                source_bucket=source_bucket,
                source_s3_key=source_s3_key,
                batch_year=valid_batch_year,
                batch_week=valid_batch_week,
                chunk_size=chunk_size,
                raw_data_schema={}
            )

            assert result["status"] == "success"
            assert result["message"] == \
            f"Successfully extracted {valid_batch_year}-W{valid_batch_week} from source s3 bucket: {source_bucket}"
            expected_df = pd.DataFrame({'ride_id':['ride_id1','ride_id2','ride_id3'],
                                                      'rideable_type':['electric_bike','classic_bike','docked_bike'],
                                                      'started_at':['2024-12-01 08:00','2024-12-01 09:00','2024-12-01 10:00'],
                                                      'ended_at':['2024-12-01 08:15','2024-12-01 09:25','2024-12-01 10:45'],
                                                      'start_station_name':['Station A','Station C','Station E'],
                                                      'start_station_id':['start_station_id1','start_station_id2','start_station_id3'],
                                                      'end_station_name':['Station B','Station D','Station F'],
                                                      'end_station_id':['end_station_id1','end_station_id2','end_station_id3'],
                                                      'start_lat':[40.7128,40.7306,40.7580],
                                                      'start_lng':[-74.0060,-73.9352,-73.9855],
                                                      'end_lat':[40.7150,40.7320,40.7600],
                                                      'end_lng':[-74.0020,-73.9300,-73.9800],
                                                      'member_casual':['member','casual','member']
                                                     })
            expected_df = expected_df.convert_dtypes(dtype_backend="numpy_nullable")
            result_df = result["data"].convert_dtypes(dtype_backend="numpy_nullable")
            pd.testing.assert_frame_equal(result_df.reset_index(drop=True), 
                                        expected_df.reset_index(drop=True))


class TestValidateRawData:
    def test_invalid_datatype_argument(self):
        """
        Tests the validate_raw_data function with an invalid argument type.
        i.e argument is not a dataframe

        Args:
            None

        Returns:
            None
        """
        invalid_raw_df_types = [
            123,
            None,
            [],
            {},
            True,
            False,
            12.34 
        ]
        for raw_df in invalid_raw_df_types:
            with pytest.raises(Exception) as exc_info:
                validate_raw_data(raw_df)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"
    
    def test_missing_required_columns(self):
        """
        Tests the validate_raw_data function with a DataFrame missing required columns.

        It should raise an Exception indicating which columns are missing.
        Args:
            None
        Returns:
            None
        """
        raw_df = pd.DataFrame({
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, -87.6288],
            "member_casual": ["member", "casual", "member"]
        })
        missing_columns = ['ride_id', 'rideable_type', 'started_at',
                           'start_station_id', 'start_station_name',
                           'end_station_id', 'end_station_name']
        
        
        with pytest.raises(Exception) as exc_info:
            
            validate_raw_data(raw_df)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": f"columns: {sorted(missing_columns)} are missing"
        }
    
    def test_identifier_column_null(self):
        """
        Tests the validate_raw_data function with a DataFrame having null values in the identifier column 'ride_id'.

        It will raise an Exception indicating that 'ride_id' contains null values.
        
        Args:
            None
        
        Returns:
            None
        """
        test_df = pd.DataFrame({
            "ride_id": [None, "ride_002", "ride_003"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_station_name": ["Station A", "Station B", "Station C"],
            "end_station_name": ["Station D", "Station E", "Station F"],
            "start_station_id": ["station_id1", "station_id2", "station_id3"],
            "end_station_id": ["station_id4", "station_id5", "station_id6"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, -87.6288],
            "member_casual": ["member", "casual", "member"]
        })
        
        with pytest.raises(Exception) as exc_info:
            validate_raw_data(test_df)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "ride_id column contains null values"
        }
    
    def test_identifier_column_not_unique(self):
        """
        Tests the validate_raw_data function with a DataFrame having duplicate values in the identifier column 'ride_id'.

        It will raise an Exception indicating that 'ride_id' contains duplicate values.
        
        Args:
            None

        Returns:
            None
        """
        test_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", "ride_001"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_station_name": ["Station A", "Station B", "Station C"],
            "end_station_name": ["Station D", "Station E", "Station F"],
            "start_station_id": ["station_id1", "station_id2", "station_id3"],
            "end_station_id": ["station_id4", "station_id5", "station_id6"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, -87.6288],
            "member_casual": ["member", "casual", "member"]
        })
        
        with pytest.raises(Exception) as exc_info:
            validate_raw_data(test_df)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column validation error occured",
            "error": "ride_id column contains duplicate values"
        }
    
    def test_incorrect_datetime_format(self):
        """
        Tests the validate_raw_data function with a DataFrame having incorrect datetime format in 'started_at' column.

        It will raise an Exception indicating that 'started_at' column has incorrect datetime format.
        
        Args:
            None

        Returns:
            None
        """
        test_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", "ride_003"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "started_at": ["2023/01/01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"], #incorrect format in first row
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_station_name": ["Station A", "Station B", "Station C"],
            "end_station_name": ["Station D", "Station E", "Station F"],
            "start_station_id": ["station_id1", "station_id2", "station_id3"],
            "end_station_id": ["station_id4", "station_id5", "station_id6"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, -87.6288],
            "member_casual": ["member", "casual", "member"]
        })

        failing_datetime_columns = ['started_at']
        with pytest.raises(Exception) as exc_info:
            validate_raw_data(test_df)
            
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column type validation error(s) occurred",
            "error": f"columns with invalid datetime format: {', '.join(failing_datetime_columns)}"
        }
    
    def test_str_column_with_non_str_values(self) -> None:
        """
        Tests the validate_raw_data function with a DataFrame having non-string values in a string column.

        It will raise an Exception indicating that the specific string column contains non-string values.
        
        Args:
            None

        Returns:
            None
        """
        test_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", 12345], #non-string value in third row
            "rideable_type": ["electric_bike", "docked_bike", 12345], #non-string value in third row
            "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_station_name": ["Station A", "Station B", "Station C"],
            "end_station_name": ["Station D", "Station E", "Station F"],
            "start_station_id": ["station_id1", "station_id2", "station_id3"],
            "end_station_id": ["station_id4", "station_id5", "station_id6"],
            "start_lat": [40.7128, 34.0522, 41.8781],
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, -87.6288],
            "member_casual": ["member", "casual", "member"]
        })

        failing_cols = ["ride_id", "rideable_type"]
        
        with pytest.raises(Exception) as exc_info:
            validate_raw_data(test_df)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column type validation error(s) occurred",
            "error": f"columns with non-string values: {', '.join(failing_cols)}"
        }
    def test_float_column_with_non_float_values(self) -> None:
        """
        Tests the validate_raw_data function with a DataFrame having non-float values in a float column.

        It will raise an Exception indicating that the specific float column contains non-float values.
        
        Args:
            None

        Returns:
            None
        """
        test_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", "ride_003"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "started_at": ["2023-01-01 09:00:00", "2023-01-01 10:00:00", "2023-01-01 11:00:00"],
            "ended_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "start_station_name": ["Station A", "Station B", "Station C"],
            "end_station_name": ["Station D", "Station E", "Station F"],
            "start_station_id": ["station_id1", "station_id2", "station_id3"],
            "end_station_id": ["station_id4", "station_id5", "station_id6"],
            "start_lat": [40.7128, "invalid_latitude", 41.8781], #non-float value in second row
            "start_lng": [-74.0060, -118.2437, -87.6298],
            "end_lat": [40.7138, 34.0532, 41.8791],
            "end_lng": [-74.0050, -118.2427, "invalid_longitude"], #non-float value in third row
            "member_casual": ["member", "casual", "member"]
        })

        failing_cols = ["start_lat", "end_lng"]
        
        with pytest.raises(Exception) as exc_info:
            validate_raw_data(test_df)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred during raw data validation"
        assert exc_info.value.args[0]["error"] == {
            "status": "error",
            "message": "column type validation error(s) occurred",
            "error": f"columns with non-float values: {', '.join(failing_cols)}"
        }


def test_requests_session_success(mock_successful_session: MagicMock) -> None:
    """
    Tests the requests_session_with_retries function to ensure it configures
    a requests Session with retry logic.

    Args:
        mock_successful_session: Fixture providing a mocked requests Session.

    Returns:
        None
    """
    session = requests_session_with_retries()

    assert isinstance(session, Session)
    assert session is mock_successful_session

    adapter = session.adapters["https://"]
    retries = adapter.max_retries

    assert retries.total == 3
    assert retries.backoff_factor == 2
    assert retries.status_forcelist == [429, 500, 502, 503, 504]
    assert retries.allowed_methods == ["GET", "POST"]


def test_requests_session_failure(mock_failing_session: MagicMock) -> None:
    """
    Tests the requests_session_with_retries function to ensure it raises an exception
    when the requests Session creation fails.

    Args:
        mock_failing_session: Fixture providing a mocked requests Session that raises an exception.

    Returns:
        None
    """
    with pytest.raises(Exception) as exc_info:
        requests_session_with_retries()

    mocked_error = exc_info.value.args[0]

    assert mocked_error["status"] == "error"
    assert mocked_error["message"] == "An error occurred while creating requests session with retries"
    assert mocked_error["error"] == "boom!"

class TestGetAddress:

    def test_invalid_argument_types(self) -> None:
        """
        Tests the get_address function with invalid argument types.

        Args:
            None

        Returns:
            None
        """
        invalid_inputs = [
            (123, -74.0060),       
            ("40.7128", -74.0060),    
            (None, "-74.0060"),
            ("40.7128", None),
            ([], -74.0060),
            ("40.7128", {}),
            (True, -74.0060),
            ("40.7128", False),
            (40.7128, "-74.0060")
        ]

        for latitude, longitude in invalid_inputs:
            with pytest.raises(Exception) as exc_info:
                get_address(latitude, longitude)

            assert exc_info.type is Exception
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": "An error occurred while fetching address from OpenStreetMap API",
                "error": "Latitude and Longitude must be string types"
            }

    @patch("src.utils.requests_session_with_retries")
    def test_success_case(self, mock_factory) -> None:
        """
        Test that get_address returns a successful response when the API
        returns HTTP 200 with a valid address payload.

        This test simulates:
            - A normal API request returning status code 200.
            - A JSON body containing a display_name field.

        Args:
            mock_factory (MagicMock): The patched requests_session_with_retries function
                that returns a mock session whose GET request simulates a successful API response.
        
        Returns:
            None
        """

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "address": {
                "city": "New York",
                "state": "NY",
                "country": "USA"
            }
        }
        mock_session.get.return_value = mock_response
        # Make the factory return this session
        mock_factory.return_value = mock_session
        latitude = "40.7128"
        longitude = "-74.0060"
        result = get_address(latitude, longitude)

        assert result == {
            "status": "success",
            "message": "Address fetched from OpenStreetMap API successfully",
            "data": "Address not found"
        }


    @patch("src.utils.requests_session_with_retries")
    def test_429_rate_limit(self, mock_factory: MagicMock) -> None:
        """
        Test that get_address raises an exception when the API returns
        HTTP 429 (rate limit exceeded).

        This test simulates:
            - A single API request returning status code 429.
            - The function raising an exception with the corresponding error payload.

        Args:
            mock_factory (MagicMock): The patched requests_session_with_retries function
                that returns a mock session whose GET request simulates a 429 rate limit response.
        """

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_session.get.return_value = mock_response

        # Make the factory return this session
        mock_factory.return_value = mock_session

        latitude = "40.7128"
        longitude = "-74.0060"

        with pytest.raises(Exception) as exc_info:
            result = get_address(latitude, longitude)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while fetching address from OpenStreetMap API"
        assert exc_info.value.args[0]['error'] == {
            "status": "error",
            "message": "Rate limit exceeded when fetching data from OpenStreetMap API",
            "error": f"{mock_response.text}"
        }

    @patch("src.utils.requests_session_with_retries")
    def test_connection_error_retry(self, mock_factory: MagicMock, 
                                    monkeypatch: MonkeyPatch) -> None:
        """
        Test that get_address retries once when a ConnectionError occurs and then succeeds.

        This test simulates:
            - The first request raising a ConnectionError.
            - The second request returning a successful response.
            - time.sleep being patched to avoid slowing down tests.

        Args:
            mock_factory (MagicMock): The patched requests_session_with_retries function that returns a mock session.
            monkeypatch (MonkeyPatch): Used to patch time.sleep to skip actual waiting.
        """

        mock_response = MagicMock()
        mock_session = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "mocked_address": "data"
        }
        mock_session.get.side_effect = [
        ConnectionError("Connection timed out"),
        mock_response
        ]
        mock_session.get.return_value = mock_response

        mock_factory.return_value = mock_session

        # Patch sleep to avoid slowing down test
        monkeypatch.setattr("src.utils.time.sleep", lambda x: None)

        resp = get_address("10", "20")
        assert resp["status"] == "success"
        assert resp["message"] == "Address fetched from OpenStreetMap API successfully"
        assert resp["data"] == "Address not found"
        assert mock_session.get.call_count == 2

    @patch("src.utils.requests_session_with_retries")
    def test_connection_error_then_failure(self, mock_factory: MagicMock, monkeypatch: MonkeyPatch) -> None:
        """
        Test that get_address retries once after a ConnectionError and raises an exception
        if the second attempt also fails.

        This test simulates:
            - The first request raising a ConnectionError.
            - The retry attempt returning a non-200 failure response.
            - time.sleep being patched to prevent delays during testing.

        Args:
            mock_factory (MagicMock): The patched requests_session_with_retries function
                used to supply mock session objects for each request attempt.
            monkeypatch (MonkeyPatch): Pytest fixture used to patch time.sleep so the retry
                does not pause the test.
        """

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session.get.side_effect = [
            ConnectionError("Connection timed out"),
            mock_response
        ] 
        mock_factory.return_value = mock_session

        monkeypatch.setattr("src.utils.time.sleep", lambda x: None)

        with pytest.raises(Exception) as exc_info:
            get_address("10", "20")

        assert mock_session.get.call_count == 2
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while making request to OpenStreetMap API"
        assert exc_info.value.args[0]['error'] == f"{mock_response.text}"

    @patch("src.utils.requests_session_with_retries")
    def test_non_200_non_429_response(self, mock_factory: MagicMock) -> None:
        """
        Test that get_address raises an exception when the API returns
        a non-200 and non-429 HTTP status code (e.g., 500 server error).

        This test simulates:
            - A single request returning a failure response such as HTTP 500.
            - No retry, since only ConnectionError triggers retry logic.

        Args:
            mock_factory (MagicMock): The patched requests_session_with_retries function
                used to return a mock session whose GET request simulates a server error.
        """
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session.get.return_value = mock_response

        latitude = "40.7128"
        longitude = "-74.0060"

        mock_factory.return_value = mock_session

        with pytest.raises(Exception) as exc_info:
            get_address(latitude, longitude)

        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while fetching address from OpenStreetMap API"
        assert exc_info.value.args[0]['error'] == {
            "status": "error",
            "message": "An error occurred while making request to OpenStreetMap API",
            "error": f"{mock_response.text}"
        }

class TestAddStationId:

    def test_add_station_id_invalid_argument_types(self) -> None:
        """
        Tests the add_station_id function with invalid argument types.

        Args:
            mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

        Returns:
            None
        """
        invalid_inputs = [
            (123, "-74.0060"),
            ("40.7128", -74.0060),    
            (None, "-74.0060"),
            ("40.7128", None),
            (40.7128, "-74.0060"),
            ("40.712", {}),
            ([], -74.0060),
            (True, -74.0060),
        ]

        for lat, lon in invalid_inputs:
            with pytest.raises(Exception) as exc_info:
                add_station_id(lat, lon)

            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while generating station ID"
            assert exc_info.value.args[0]['error'] == "Latitude and Longitude must be string types"

    @patch("src.utils.gen_hash_key_station_id")
    def test_add_station_id_success(self, 
                                    mock_gen_hash_key: MagicMock) -> None:
        """
        Tests the add_station_id function with valid inputs.
        It verifies that the function returns the expected station ID.

        Args:
            mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

        Returns:
            None
        """
        latitude = "40.7128"
        longitude = "-74.0060"
        expected_hash_key_response = {
            "status": "success",
            "message": "Hash key generated successfully",
            "hash_key": "station_12345"
        }

        mock_gen_hash_key.return_value = expected_hash_key_response
        expected_station_id = mock_gen_hash_key.return_value['hash_key']

        result = add_station_id(latitude, longitude)

        assert result['status'] == "success"
        assert result['message'] == "Station ID generated successfully"
        assert result['data'] == expected_station_id
        mock_gen_hash_key.assert_called_once_with(latitude, longitude)

    @patch("src.utils.gen_hash_key_station_id")
    def test_add_station_id_gen_hash_key_failure(self, 
                                                 mock_gen_hash_key: MagicMock) -> None:
        """
        Tests the add_station_id function when gen_hash_key_station_id fails.
        It verifies that the function raises an exception with the expected error message.

        Args:
            mock_gen_hash_key (MagicMock): The patched gen_hash_key_station_id function.

        Returns:
            None
        """
        latitude = "40.7128"
        longitude = "-74.0060"

        mocked_gen_hash_key_error = {
            "status": "error",
            "message": "Unable to generate hash key for station ID",
            "error": "hash key generation error"
        }

        mock_gen_hash_key.side_effect = Exception(
            mocked_gen_hash_key_error
        )
        expected_error_response = {
            "status": "error",
            "message": "An error occurred while generating station ID",
            "error": mocked_gen_hash_key_error
        }

        with pytest.raises(Exception) as exc_info:
            add_station_id(latitude, longitude)

        assert exc_info.type is Exception
        assert exc_info.value.args[0] == expected_error_response
        mock_gen_hash_key.assert_called_once_with(latitude, longitude)


class TestCleanRawData:
    """
    Tests for the clean_raw_data function.
    """

    def test_invalid_argument_type(self) -> None:
        """
        Tests the clean_raw_data function with an invalid argument type.
        i.e argument is not a dataframe

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

        for raw_df in invalid_raw_df_types:
            with pytest.raises(Exception) as exc_info:
                clean_raw_data(raw_df)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while cleaning raw data"
        assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"
    
    def test_deduplication_and_standardization(self) -> None:
        """
        Tests the clean_raw_data function for deduplication and standardization of column names(all lowercase).
        It tests the success case where the input DataFrame contains duplicate rows and string type columns with inconsistent casing and leading/trailing spaces.
        It verifies that duplicate rows based on 'ride_id' are removed and string type columns are stripped and converted to lowercase.

        Args:
            None
        
        Returns:
            None
        """
        raw_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", "ride_001", "ride_003"], #duplicate ride_id "ride_001"
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike", "electric_bike"],
            "start_station_name": [" Station A ", "Station B", " station a ", "Station C"],
            "start_station_id": ["ST001", "ST002", "ST001", "ST003"],
            "end_station_name": [" Station X ", "Station Y", " station x ", "Station Z"],
            "end_station_id": ["STX01", "STY02", "STX01", "STZ03"],
            "member_casual": ["Member", "Casual", "Member", "Member"]
        })

        expected_cleaned_df = pd.DataFrame({
            "ride_id": ["ride_001", "ride_002", "ride_003"],
            "rideable_type": ["electric_bike", "docked_bike", "electric_bike"],
            "start_station_name": ["station a", "station b", "station c"],
            "start_station_id": ["st001", "st002", "st003"],
            "end_station_name": ["station x", "station y", "station z"],
            "end_station_id": ["stx01", "sty02", "stz03"],
            "member_casual": ["member", "casual", "member"]
        }).reset_index(drop=True)

        result = clean_raw_data(raw_df)
        df_result = result['data'].reset_index(drop=True)
        pd.testing.assert_frame_equal(df_result,
                                       expected_cleaned_df)
        assert df_result.duplicated().sum() == 0
        assert df_result.columns.str.islower().all()
        assert (df_result.columns == df_result.columns.str.strip()).all()


class TestImputeMissingStationIDs:
    """
    Tests for the impute_missing_station_ids function.
    """ 
    def test_invalid_argument_type(self) -> None:
        """
        Tests the impute_missing_station_ids function with an invalid argument type.
        i.e argument is not a dataframe

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

        for raw_df in invalid_raw_df_types:
            with pytest.raises(Exception) as exc_info:
                impute_missing_station_ids(raw_df)
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]["status"] == "error"
        assert exc_info.value.args[0]["message"] == "An error occurred while imputing missing station IDs"
        assert exc_info.value.args[0]["error"] == "Input raw_df must be a pandas DataFrame"


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
        invalid_raw_s3_config_types = [ 
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

        for s3_config in invalid_raw_s3_config_types:
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
        valid_raw_s3_config = {
            "bucket_name": "test-bucket",
            "aws_access_key_id": "TESTACCESSKEY",
            "aws_secret_access_key": "TESTSECRETKEY",
            "region_name": "us-east-1"
        }

        for processed_s3_key in invalid_processed_s3_key_types:
            with pytest.raises(Exception) as exc_info:
                validate_processed_data(valid_raw_s3_config, processed_s3_key) #processed_s3_key must be a string
        
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


class TestListPartitionsInputValidation:
    """
    This class tests the "list_partitions" function for invalid input types
    """

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        

    def test_bucket_not_string_raises(self) -> None:
        with pytest.raises(Exception) as exc_info:
            invalid_bucket_types = [123, None, [], {}, True, False, 12.34, (1, 2, 3)]
            for non_string_bucket in invalid_bucket_types:
                list_partitions(non_string_bucket, self.valid_raw_bucket_prefix,
                                 self.valid_access_key, self.valid_secret_key)
            assert exc_info.value.args[0] == {
                "status": "error",
                "message": f"An error occurred while listing partitions in bucket: {non_string_bucket} with prefix: {self.valid_raw_bucket_prefix}",
                "error": "bucket argument must be a string"
            }

    def test_prefix_not_string_raises(self) -> None:
        with pytest.raises(Exception) as exc_info:
            invalid_raw_bucket_prefix_types = [123, None, [], {}, True, False, 12.34, (1, 2, 3)]
            for non_string_prefix in invalid_raw_bucket_prefix_types:
                list_partitions(self.valid_raw_bucket, non_string_prefix, 
                                self.valid_access_key, self.valid_secret_key)

            assert exc_info.value.args[0] == {
                    "status": "error",
                    "message": f"An error occurred while listing partitions in bucket: {self.valid_raw_bucket} with prefix: {self.valid_raw_bucket_prefix}",
                    "error": "raw_bucket_prefix argument must be a string"
                }
    
    def test_access_key_not_string_raises(self) -> None:
        
        with pytest.raises(Exception) as exc_info:
            invalid_access_key_types = [123, None, [], {}, True, False, 12.34, (1, 2, 3)]
            for non_string_access_key in invalid_access_key_types:
                list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, 
                                non_string_access_key, self.valid_secret_key)

            assert exc_info.value.args[0] == {
                    "status": "error",
                    "message": f"An error occurred while listing partitions in bucket: {self.valid_raw_bucket} with prefix: {self.valid_raw_bucket_prefix}",
                    "error": "access_key argument must be a string"
                }

    def test_access_key_not_string_raises(self) -> None:
        
        with pytest.raises(Exception) as exc_info:
            invalid_access_key_types = [123, None, [], {}, True, False, 12.34, (1, 2, 3)]
            for non_string_access_key in invalid_access_key_types:
                list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, 
                                non_string_access_key, self.valid_secret_key)

            assert exc_info.value.args[0] == {
                    "status": "error",
                    "message": f"An error occurred while listing partitions in bucket: {self.valid_raw_bucket} with prefix: {self.valid_raw_bucket_prefix}",
                    "error": "access_key argument must be a string"
                }

    def test_secret_key_not_string_raises(self) -> None:
        with pytest.raises(Exception) as exc_info:
            invalid_secret_key_types = [123, None, [], {}, True, False, 12.34, (1, 2, 3)]

            for non_string_access_key in invalid_secret_key_types:
                list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, 
                                    non_string_access_key, self.valid_secret_key)

            assert exc_info.value.args[0] == {
                    "status": "error",
                    "message": f"An error occurred while listing partitions in bucket: {self.valid_raw_bucket} with prefix: {self.valid_raw_bucket_prefix}",
                    "error": "secret_key argument must be a string"
                }

    def test_all_none_raises(self):
        """All None args should still raise — not silently pass."""
        with pytest.raises(Exception) as exc_info:
            list_partitions(None, None, None, None)
        
        assert exc_info.value.args[0] == {'status': 'error', 
                                          'message': 'An error occurred while listing partitions in bucket: None with folder: None', 
                                          'error': 'bucket argument must be a string'}




class TestListPartitionsValidPath:

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        

    @patch('src.utils.boto3.client')
    def test_single_partition_single_file(self,
                                        mock_boto3_client: MagicMock,
                                        mimick_s3_page: dict):
        keys = [
            "trips/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv"
        ]
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket, 
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        
        assert result["status"] == "success"
        partitions = result["partitions"]
        assert ("2022", "48") in partitions
        assert len(partitions[("2022", "48")]) == 1
        assert partitions[("2022", "48")][0][0] == "2022-11-28T10-00-00Z"
       
    @patch('src.utils.boto3.client')
    def test_multiple_reruns_same_week_sorted_ascending(self, 
                                                        mock_boto3_client: MagicMock,
                                                        mimick_s3_page: dict):
        """
        Core idempotency requirement: multiple ingestion_ts for the same
        (year, week) must be sorted ascending so [-1] gives latest.
        """
        keys = [
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T22-00-00Z/trips.csv",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T15-30-00Z/trips.csv",
        ]
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        # result = list_partitions(VALID_BUCKET, VALID_PREFIX, VALID_ACCESS_KEY, VALID_SECRET_KEY)
        result = list_partitions(self.valid_raw_bucket, 
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
            
        ts_list = [ts for ts, _ in result["partitions"][("2022", "48")]]
        assert ts_list == sorted(ts_list), "ingestion_ts entries must be sorted ascending"
        assert ts_list[-1] == "2022-11-28T22-00-00Z", "Latest rerun must be last"


    @patch('src.utils.boto3.client')
    def test_multiple_weeks_partitioned_correctly(self,
                                                mock_boto3_client,
                                                mimick_s3_page: dict):
        
        keys = [
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv",
            "bikeshare/year=2022/week=49/ingestion_ts=2022-12-05T10-00-00Z/trips.csv",
            "bikeshare/year=2023/week=1/ingestion_ts=2023-01-02T10-00-00Z/trips.csv",
        ]

        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]
        
        result = list_partitions(self.valid_raw_bucket, 
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        
        partitions = result["partitions"]

        assert ("2022", "48") in partitions
        assert ("2022", "49") in partitions
        assert ("2023", "1") in partitions
        assert len(partitions) == 3

    @patch('src.utils.boto3.client')
    def test_pagination_across_multiple_pages(self,
                                            mock_boto3_client,
                                            mimick_s3_page: dict):
        """
        S3 paginates at 1000 objects. Ensure all pages are consumed.
        """
        page1 = mimick_s3_page([
            "bikeshare/year=2022/week=47/ingestion_ts=2022-11-21T10-00-00Z/trips.csv"
        ])
        page2 = mimick_s3_page([
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv"
        ])
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [page1, page2]

        result = list_partitions(self.valid_raw_bucket, 
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)

        assert ("2022", "47") in result["partitions"]
        assert ("2022", "48") in result["partitions"]

    @patch('src.utils.boto3.client')
    def test_success_response_shape(self, mock_boto3_client):
        """Contract test — callers depend on this exact shape."""
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        result = list_partitions(self.valid_raw_bucket, 
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)

        assert "status" in result
        assert "message" in result
        assert "partitions" in result
        assert result["status"] == "success"
        assert isinstance(result["partitions"], dict)


# ---------------------------------------------------------------------------
# 3. Edge Cases — S3 Object Filtering
# ---------------------------------------------------------------------------

class TestListPartitionsRawS3ObjectFiltering:

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        

    @patch('src.utils.boto3.client')
    def test_non_csv_files_are_ignored(self, 
                                       mock_boto3_client: MagicMock,
                                       mimick_s3_page: dict):
        """Parquet, json, _SUCCESS markers etc. must be filtered out."""
        keys = [
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.parquet",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/_SUCCESS",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.json",
        ]
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket,
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        assert result["partitions"] == {}

    @patch('src.utils.boto3.client')
    def test_keys_not_matching_partition_pattern_ignored(self, 
                                                         mock_boto3_client: MagicMock,
                                                         mimick_s3_page: dict):
        """Stray files at the root or with wrong structure must be skipped."""
        keys = [
            "bikeshare/README.csv",
            "bikeshare/2022/48/trips.csv",            # missing partition= format
            "bikeshare/year=2022/trips.csv",           # missing week and ingestion_ts
        ]
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket,
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        assert result["partitions"] == {}

    @patch('src.utils.boto3.client')
    def test_empty_bucket_prefix_returns_empty_partitions(self, mock_boto3_client: MagicMock, 
                                                          make_empty_page: dict):
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        result = list_partitions(self.valid_raw_bucket,
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)

        assert result["status"] == "success"
        assert result["partitions"] == {}

    @patch('src.utils.boto3.client')
    def test_page_with_no_contents_key_handled(self,
                                                mock_boto3_client: MagicMock,
                                                ):
        """
        S3 returns pages without a 'Contents' key when the prefix is empty.
        page.get('Contents', []) must not raise.
        """
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]   # no 'Contents' key at all

        result = list_partitions(self.valid_raw_bucket,
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        assert result["partitions"] == {}

    @patch('src.utils.boto3.client')
    def test_mixed_valid_and_invalid_keys(self, 
                                            mock_boto3_client: MagicMock,
                                            mimick_s3_page: dict):
        """Only valid partition paths should be picked up."""
        keys = [
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv",  # valid
            "bikeshare/README.csv",                                                        # invalid
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.json",  # wrong ext
        ]
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket,
                                 self.valid_raw_bucket_prefix, 
                                    self.valid_access_key,
                                      self.valid_secret_key)
        
        assert len(result["partitions"][("2022", "48")]) == 1



class TestListPartitionsAWSErrors:

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        

    @patch('src.utils.boto3.client')
    def test_invalid_credentials_raises(self, mock_boto3_client):
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "InvalidClientTokenId", "Message": "Invalid credentials"}},
            "ListObjectsV2"
        )
        with pytest.raises(Exception) as exc_info:
            list_partitions(self.valid_raw_bucket,
                            self.valid_raw_bucket_prefix,
                            self.valid_access_key,
                            self.valid_secret_key)
        assert "error" in str(exc_info.value).lower() or "InvalidClientTokenId" in str(exc_info.value)

    @patch('src.utils.boto3.client')
    def test_bucket_not_found_raises(self, mock_boto3_client: MagicMock):
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}},
            "ListObjectsV2"
        )
        with pytest.raises(Exception) as exc_info:
            list_partitions(self.valid_raw_bucket,
                            self.valid_raw_bucket_prefix,
                            self.valid_access_key,
                            self.valid_secret_key)
            
        assert "error" in str(exc_info.value).lower()

    @patch('src.utils.boto3.client')
    def test_access_denied_raises(self,
                            mock_boto3_client: MagicMock):
        
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "ListObjectsV2"
        )
        with pytest.raises(Exception):
            list_partitions(self.valid_raw_bucket,
                            self.valid_raw_bucket_prefix,
                            self.valid_access_key,
                            self.valid_secret_key)

    @patch('src.utils.boto3.client')
    def test_no_credentials_error_raises(self, mock_boto3_client: MagicMock):
        mock_boto3_client.return_value.get_paginator.side_effect = NoCredentialsError()
        with pytest.raises(Exception):
            list_partitions(self.valid_raw_bucket,
                             self.valid_raw_bucket_prefix,
                               self.valid_access_key,
                                 self.valid_secret_key)

    @patch('src.utils.boto3.client')
    def test_error_response_shape(self, mock_boto3_client: MagicMock):
        """
        Callers catching the raised Exception need a consistent error shape.
        The raised Exception must carry status/message/error keys.
        """
        mock_paginator = MagicMock()
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket missing"}},
            "ListObjectsV2"
        )
        with pytest.raises(Exception) as exc_info:
            list_partitions(
                self.valid_raw_bucket,
                self.valid_raw_bucket_prefix,
                self.valid_access_key,
                self.valid_secret_key
            )

        # The raised exception wraps a dict — evaluate it
        raised = ast.literal_eval(str(exc_info.value))
        assert raised["status"] == "error"
        assert "message" in raised
        assert "error" in raised


# # ---------------------------------------------------------------------------
# # 5. Boto3 Client Construction
# # ---------------------------------------------------------------------------

class TestClientConstruction:

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        
    @patch('src.utils.boto3.client')
    def test_boto3_client_called_with_correct_credentials(self, mock_boto3_client):
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        # with patch("your_module.boto3.client") as mock_constructor:
        #     mock_s3 = MagicMock()
        #     mock_constructor.return_value = mock_s3
        #     mock_s3.get_paginator.return_value = mock_paginator

        list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, self.valid_access_key, self.valid_secret_key)

        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id=self.valid_access_key,
            aws_secret_access_key=self.valid_secret_key
        )

    @patch('src.utils.boto3.client')
    def test_paginator_called_with_correct_bucket_and_prefix(self, mock_boto3_client):
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, self.valid_access_key, self.valid_secret_key)

        mock_paginator.paginate.assert_called_once_with(
            Bucket=self.valid_raw_bucket,
            Prefix=self.valid_raw_bucket_prefix
        )


# ---------------------------------------------------------------------------
# 6. Partition Sorting — Determinism
# ---------------------------------------------------------------------------

class TestListPartitionsSorting:

    @pytest.fixture(autouse=True)
    def _setup_creds(self, aws_credentials: dict) -> None:
        """
        This helper method extracts valid AWS credentials from the provided fixture for use in the tests.

        Args:
            aws_credentials (dict): A dictionary containing AWS credentials, expected to have keys "aws_access_key" and "aws_secret_key".

        Returns:
            None
        """
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        

    @patch('src.utils.boto3.client')
    def test_sort_is_lexicographic_on_iso_timestamp(self, mock_boto3_client: MagicMock,
                                                    mimick_s3_page: dict):
        """
        ISO timestamps with dashes instead of colons (T10-00-00Z) sort
        correctly lexicographically — validate this assumption holds.
        """
        keys = [
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T23-59-59Z/trips.csv",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T00-00-01Z/trips.csv",
            "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T12-00-00Z/trips.csv",
        ]
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, self.valid_access_key, self.valid_secret_key)
        ts_list = [ts for ts, _ in result["partitions"][("2022", "48")]]

        assert ts_list[0] == "2022-11-28T00-00-01Z"
        assert ts_list[-1] == "2022-11-28T23-59-59Z"

    @patch('src.utils.boto3.client')
    def test_sort_across_year_boundary(self, mock_boto3_client: MagicMock,
                                       mimick_s3_page: dict):
        keys = [
            "bikeshare/year=2022/week=52/ingestion_ts=2022-12-31T23-00-00Z/trips.csv",
            "bikeshare/year=2022/week=52/ingestion_ts=2022-12-30T08-00-00Z/trips.csv",
        ]
        mock_paginator = MagicMock()
        mock_boto3_client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [mimick_s3_page(keys)]

        result = list_partitions(self.valid_raw_bucket, self.valid_raw_bucket_prefix, self.valid_access_key, self.valid_secret_key)
        ts_list = [ts for ts, _ in result["partitions"][("2022", "52")]]
        assert ts_list[-1] == "2022-12-31T23-00-00Z"


class TestYieldEventStream:
    
    @pytest.fixture(autouse=True)
    def mimick_s3_body(self) -> None:
        def make_s3_body(csv_content: str):
            """Builds the mock S3 object body that mimics boto3's StreamingBody."""
            mock_body = MagicMock()
            mock_body.read.return_value = csv_content.encode("utf-8")
            return mock_body
        return make_s3_body

    @pytest.fixture(autouse=True)
    def setup(self, aws_credentials):
        self.valid_access_key = aws_credentials["aws_access_key"]
        self.valid_secret_key = aws_credentials["aws_secret_key"]
        self.valid_raw_bucket = aws_credentials["raw_bucket"]
        self.valid_raw_bucket_prefix = aws_credentials["raw_bucket_folder"]
        self.chunk_size = 100000
        self.delay_seconds = 2
        self.SAMPLE_CSV = """ride_id,member_casual,started_at,ended_at,start_station_name
abc123,casual,2022-11-28 00:05:00,2022-11-28 01:00:00,Main St
def456,member,2022-11-28 08:00:00,2022-11-28 08:30:00,Park Ave
"""
        self.SINGLE_PARTITION = {
            ("2022", "48"): [
                ("2022-11-28T10-00-00Z",
                "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv")
            ]
        }

        self.MULTI_RERUN_PARTITION = {
            ("2022", "48"): [
                ("2022-11-28T08-00-00Z",
                "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T08-00-00Z/trips.csv"),
                ("2022-11-28T10-00-00Z",
                "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv"),  # latest
            ]
        }

        self.MULTI_WEEK_PARTITION = {
            ("2022", "47"): [
                ("2022-11-21T10-00-00Z",
                "bikeshare/year=2022/week=47/ingestion_ts=2022-11-21T10-00-00Z/trips.csv")
            ],
            ("2022", "48"): [
                ("2022-11-28T10-00-00Z",
                "bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv")
            ],
        }


class TestInputValidation(TestYieldEventStream):

    invalid_non_string_types = [123, 123.5, {}, (), [], None]
    
    def _call(self,
                **overrides) -> Generator[dict, None, None]:
                
        kwargs = dict(
            raw_bucket=self.valid_raw_bucket,
            raw_bucket_prefix=self.valid_raw_bucket_prefix,
            aws_access_key=self.valid_access_key,
            aws_secret_key=self.valid_secret_key,
            chunk_size=self.chunk_size,
            delay_seconds=self.delay_seconds
        )
        kwargs.update(overrides)
        # Drain the generator to trigger the exception
        return list(yield_event_stream(**kwargs))
    

    def test_raw_bucket_not_string_raises(self):
        
        with pytest.raises(Exception) as exc_info:
            for non_string in self.invalid_non_string_types:
                self._call(raw_bucket=non_string)
        
                assert exc_info.value.args[0] == {'status': 'error', 
                                                'message': 'An error occurred while yielding event stream from S3', 
                                                'error': 'raw_bucket argument must be a string'}

    def test_raw_bucket_prefix_not_string_raises(self):
        with pytest.raises(Exception) as exc_info:
            for non_string in self.invalid_non_string_types:
                self._call(raw_bucket_prefix=non_string)
                assert exc_info.value.args[0] == {'status': 'error', 
                                                'message': 'An error occurred while yielding event stream from S3', 
                                                'error': 'raw_bucket_prefix argument must be a string'}

    def test_access_key_not_string_raises(self):
        with pytest.raises(Exception) as exc_info:
            for non_string in self.invalid_non_string_types:
                self._call(aws_access_key=non_string)
                assert exc_info.value.args[0] == {'status': 'error', 
                                                'message': 'An error occurred while yielding event stream from S3', 
                                                'error': 'aws_access_key argument must be a string'}

    def test_secret_key_not_string_raises(self):
        with pytest.raises(Exception) as exc_info:
            for non_string in self.invalid_non_string_types:
                self._call(aws_secret_key=["oops"])
                assert exc_info.value.args[0] == {'status': 'error', 
                                                'message': 'An error occurred while yielding event stream from S3', 
                                                'error': 'aws_secret_key argument must be a string'}

    def test_chunk_size_not_int_raises(self):
        with pytest.raises(Exception) as exc_info:
            self._call(chunk_size="100")
        assert exc_info.value.args[0] == {'status': 'error', 
                                        'message': 'An error occurred while yielding event stream from S3', 
                                        'error': 'chunk_size argument must be a positive integer'}

    def test_chunk_size_zero_raises(self):
        with pytest.raises(Exception) as exc_info:
            self._call(chunk_size=0)
        assert exc_info.value.args[0] == {'status': 'error', 
                                        'message': 'An error occurred while yielding event stream from S3', 
                                        'error': 'chunk_size argument must be a positive integer'}

    def test_chunk_size_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            self._call(chunk_size=-5)
        assert exc_info.value.args[0] == {'status': 'error', 
                                        'message': 'An error occurred while yielding event stream from S3', 
                                        'error': 'chunk_size argument must be a positive integer'}

    def test_delay_seconds_not_int_raises(self):
        with pytest.raises(Exception) as exc_info:
            self._call(delay_seconds=0.5)
        assert exc_info.value.args[0] == {'status': 'error', 
                                        'message': 'An error occurred while yielding event stream from S3', 
                                        'error': 'delay_seconds argument must be a non-negative integer'}

    def test_delay_seconds_negative_raises(self):
        with pytest.raises(Exception) as exc_info:
            self._call(delay_seconds=-1)
        assert exc_info.value.args[0] == {'status': 'error', 
                                        'message': 'An error occurred while yielding event stream from S3', 
                                        'error': 'delay_seconds argument must be a non-negative integer'}

    def test_delay_seconds_zero_is_valid(self):
        """Zero delay is explicitly allowed — non-negative, not positive."""
        with patch("src.utils.boto3.client") as mock_client, \
                patch("src.utils.list_partitions") as mock_lp, \
                patch("src.utils.time.sleep"):

            mock_lp.return_value = {"partitions": {}}
            events = list(yield_event_stream(
                raw_bucket=self.valid_raw_bucket,
                raw_bucket_prefix=self.valid_raw_bucket_prefix,
                aws_access_key=self.valid_access_key,
                aws_secret_key=self.valid_secret_key,
                chunk_size=self.chunk_size,
                delay_seconds=self.delay_seconds
            ))

            assert events == []

 
class TestHappyPath(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_yields_correct_number_of_events(
        self, mock_boto3, mock_lp, mock_sleep,
        mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(csv_content=self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))  

        # SAMPLE_CSV has 2 data rows
        assert len(events) == 2

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_event_contains_lineage_metadata(
        self, 
        mock_boto3, 
        mock_lp, 
        mock_sleep,
        mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))

        for event in events:
            assert "_year" in event
            assert "_week" in event
            assert "_ingestion_ts" in event

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_lineage_values_are_correct(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))

        assert all(e["_year"] == "2022" for e in events)
        assert all(e["_week"] == "48" for e in events)
        assert all(e["_ingestion_ts"] == "2022-11-28T10-00-00Z" for e in events)

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_csv_fields_present_in_event(
        self, mock_boto3, mock_lp, mock_sleep,
        mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))

        assert events[0]["ride_id"] == "abc123"
        assert events[0]["member_casual"] == "casual"
        assert events[1]["ride_id"] == "def456"

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_is_a_generator(self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body):
        """Must be a generator — callers rely on lazy evaluation."""
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        result = yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        )
        assert isinstance(result, Generator)

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_empty_partitions_yields_nothing(
        self, mock_boto3, mock_lp, mock_sleep,
        mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": {}}

        events = list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))
        assert events == []

    # -----------------------------------------------------------------------
    # 3. Idempotency — Latest Rerun Only
    # -----------------------------------------------------------------------

class TestIdempotency(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_only_latest_ingestion_ts_is_read(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        """
        With 2 reruns for week=48, get_object must only be called ONCE
        using the latest ingestion_ts key.
        """
        mock_lp.return_value = {"partitions": self.MULTI_RERUN_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        list(yield_event_stream(
            self.valid_raw_bucket,
            self.valid_raw_bucket_prefix,
            self.valid_access_key,
            self.valid_secret_key,
            self.chunk_size,
            self.delay_seconds
        ))

        mock_s3.get_object.assert_called_once_with(
            Bucket=self.valid_raw_bucket,
            Key="bikeshare/year=2022/week=48/ingestion_ts=2022-11-28T10-00-00Z/trips.csv"
        )

        @patch("src.utils.time.sleep")
        @patch("src.utils.list_partitions")
        @patch("src.utils.boto3.client")
        def test_ingestion_ts_in_event_matches_latest(
            self, mock_boto3, mock_lp, mock_sleep,
            mimick_s3_body
        ):
            mock_lp.return_value = {"partitions": self.MULTI_RERUN_PARTITION}
            mock_s3 = mock_boto3.return_value
            mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

            events = list(yield_event_stream(
                self.valid_raw_bucket,
                self.valid_raw_bucket_prefix,
                self.valid_access_key,
                self.valid_secret_key,
                self.chunk_size,
                self.delay_seconds
            ))

            # All events must carry the LATEST ts, not the earlier one
            assert all(e["_ingestion_ts"] == "2022-11-28T10-00-00Z" for e in events)


        @patch("src.utils.time.sleep")
        @patch("src.utils.list_partitions")
        @patch("src.utils.boto3.client")
        def test_multiple_weeks_each_read_once(
            self, mock_boto3, mock_lp, mock_sleep
        ):
            mock_lp.return_value = {"partitions": self.MULTI_WEEK_PARTITION}
            mock_s3 = mock_boto3.return_value
            mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

            list(yield_event_stream(
                self.valid_raw_bucket,
                self.valid_raw_bucket_prefix,
                self.valid_access_key,
                self.valid_secret_key,
                self.chunk_size,
                self.delay_seconds
            ))

            # get_object called once per week partition
            assert mock_s3.get_object.call_count == 2


class TestStreamingDelay(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_sleep_called_once_per_row(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        """sleep must fire after EVERY row, not per chunk or per file."""
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}
    
        list(yield_event_stream(
            self.valid_raw_bucket, self.valid_raw_bucket_prefix,
            self.valid_access_key, self.valid_secret_key,
            self.chunk_size, self.delay_seconds
        ))

        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2)

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_zero_delay_still_calls_sleep(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        """Even delay=0 must call sleep(0) — consistent contract."""
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}
        
        list(yield_event_stream(
            self.valid_raw_bucket, self.valid_raw_bucket_prefix,
            self.valid_access_key, self.valid_secret_key,
            self.chunk_size, self.delay_seconds
        ))

        mock_sleep.assert_called_with(2)


class TestChunking(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_chunk_size_one_yields_one_row_at_a_time(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket, self.valid_raw_bucket_prefix,
                self.valid_access_key, self.valid_secret_key,
                self.chunk_size, self.delay_seconds
        ))
        assert len(events) == 2

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_chunk_size_larger_than_file_still_yields_all_rows(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.return_value = {"Body": mimick_s3_body(self.SAMPLE_CSV)}

        events = list(yield_event_stream(
            self.valid_raw_bucket, self.valid_raw_bucket_prefix,
                self.valid_access_key, self.valid_secret_key,
                self.chunk_size, self.delay_seconds
        ))
        assert len(events) == 2


class TestAWSErrors(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_get_object_client_error_raises(
        self, mock_boto3, mock_lp, mock_sleep
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject"
        )

        with pytest.raises(Exception):
            list(yield_event_stream(
                self.valid_raw_bucket, self.valid_raw_bucket_prefix,
                self.valid_access_key, self.valid_secret_key,
                self.chunk_size, self.delay_seconds
            ))

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_access_denied_raises(
        self, mock_boto3, mock_lp, mock_sleep
    ):
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "GetObject"
        )
        with pytest.raises(Exception):
            list(yield_event_stream(
                self.valid_raw_bucket, self.valid_raw_bucket_prefix,
        self.valid_access_key, self.valid_secret_key,
        self.chunk_size, self.delay_seconds
            ))

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_list_partitions_failure_propagates(
        self, mock_boto3, mock_lp, mock_sleep
    ):
        """If list_partitions itself raises, yield_event_stream must re-raise."""
        mock_lp.side_effect = Exception({
            "status": "error",
            "message": "Partition listing failed",
            "error": "NoSuchBucket"
        })
        with pytest.raises(Exception):
            list(yield_event_stream(
                self.valid_raw_bucket, self.valid_raw_bucket_prefix,
        self.valid_access_key, self.valid_secret_key,
        self.chunk_size, self.delay_seconds
            ))

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_error_response_shape(
        self, mock_boto3, mock_lp, mock_sleep
    ):
        """Raised exception must carry the standard error envelope."""
        mock_lp.return_value = {"partitions": self.SINGLE_PARTITION}
        mock_s3 = mock_boto3.return_value
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Missing"}},
            "GetObject"
        )
        with pytest.raises(Exception) as exc_info:
            list(yield_event_stream(
                self.valid_raw_bucket, self.valid_raw_bucket_prefix,
        self.valid_access_key, self.valid_secret_key,
        self.chunk_size, self.delay_seconds
            ))

        raised = ast.literal_eval(str(exc_info.value))
        assert raised["status"] == "error"
        assert "message" in raised
        assert "error" in raised


class TestClientConstruction(TestYieldEventStream):

    @patch("src.utils.time.sleep")
    @patch("src.utils.list_partitions")
    @patch("src.utils.boto3.client")
    def test_boto3_called_with_correct_credentials(
        self, mock_boto3, mock_lp, mock_sleep, mimick_s3_body
    ):
        mock_lp.return_value = {"partitions": {}}

        list(yield_event_stream(
            self.valid_raw_bucket, self.valid_raw_bucket_prefix,
            self.valid_access_key, self.valid_secret_key,
            self.chunk_size, self.delay_seconds
        ))

        mock_boto3.assert_called_once_with(
            "s3",
            aws_access_key_id=self.valid_access_key,
            aws_secret_access_key=self.valid_secret_key
        )


class TestSendSlackAlert:
    """
    Tests for the send_slack_alert function.
    """

    def test_invalid_message_argument_type(
        self
    ) -> None:
        """
        Tests the send_slack_alert function with an invalid message argument type.
        Here, the "message" argument must be a string to pass validation.

        Args:
            None

        Returns:
            None

        """
        invalid_message_types = [ 
            123,
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_channel = "#test-channel"

        for message in invalid_message_types:
            with pytest.raises(Exception) as exc_info:
                send_slack_alert(message=message, 
                                channel=valid_channel,
                                oauth_token="xoxb-test-token",
                                slack_bot="test-bot") 
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
        assert exc_info.value.args[0]['error'] == "message argument must be a string"

    def test_invalid_channel_argument_type(
        self
    ) -> None:
        """
        Tests the send_slack_alert function with an invalid channel argument type.
        Here, the "channel" argument must be a string to pass validation.

        Args:
            None

        Returns:
            None
        """
        invalid_channel_types = [ 
            123,
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_message = "Test Slack alert message"

        for channel in invalid_channel_types:
            with pytest.raises(Exception) as exc_info:
                send_slack_alert(message=valid_message, 
                                channel=channel,
                                oauth_token="xoxb-test-token",
                                slack_bot="test-bot") 
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
        assert exc_info.value.args[0]['error'] == "channel argument must be a string"
    

    def test_invalid_oauth_token_argument_type(
        self
    ) -> None:
        
        """
        Tests the send_slack_alert function with an invalid oauth_token argument type.
        Here, the "oauth_token" argument must be a string to pass validation.

        Args:
            None

        Returns:
            None
        """
        invalid_oauth_token_types = [ 
            123,
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_message = "Test Slack alert message"
        valid_channel = "#test-channel"

        for oauth_token in invalid_oauth_token_types:
            with pytest.raises(Exception) as exc_info:
                send_slack_alert(message=valid_message, 
                                channel=valid_channel,
                                oauth_token=oauth_token,
                                slack_bot="test-bot") 
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
        assert exc_info.value.args[0]['error'] == "oauth_token argument must be a string"

    def test_invalid_slack_bot_argument_type(
        self
    ) -> None:
        """
        Tests the send_slack_alert function with an invalid slack_bot argument type.
        Here, the "slack_bot" argument must be a string to pass validation.

        Args:
            None

        Returns:
            None
        """
        invalid_slack_bot_types = [ 
            123,
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            None,
            [],
            {},
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        valid_message = "Test Slack alert message"
        valid_channel = "#test-channel"
        valid_oauth_token = "xoxb-test-token"

        for slack_bot in invalid_slack_bot_types:
            with pytest.raises(Exception) as exc_info:
                send_slack_alert(message=valid_message, 
                                channel=valid_channel,
                                oauth_token=valid_oauth_token,
                                slack_bot=slack_bot) 
        
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while sending alert to Slack"
        assert exc_info.value.args[0]['error'] == "slack_bot argument must be a string"


    @patch('src.utils.WebClient')
    def test_send_slack_alert_failure(self, 
                                      mock_factory: MagicMock) -> None:
        """
        Tests the send_slack_alert function for failed Slack alert sending.
        It verifies that the function correctly handles a failure response from Slack API.

        Args:
            mock_factory: Mocked WebClient factory.

        Returns:
            None
        """
        mock_session = MagicMock()
        mock_session.chat_postMessage.side_effect = Exception("The request to the slack API failed.")
        mock_factory.return_value = mock_session

        message = "Test Slack alert message"
        channel = "#test-channel"
        with pytest.raises(Exception) as exc_info:
            send_slack_alert(message=message, 
                            channel="#test-channel",
                            oauth_token="xoxb-test-token",
                            slack_bot="test-bot")
        
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == f"An error occurred while sending alert to Slack"
        assert exc_info.value.args[0]['error'] == 'The request to the slack API failed.'
      

    @patch('src.utils.WebClient')
    def test_send_slack_alert_success(self, mock_factory: MagicMock) -> None:
        """
        
        Tests the send_slack_alert function for successful Slack alert sending.
        It verifies that the function correctly sends a Slack alert and returns a success response.
        
        Args:
            mock_factory: Mocked WebClient factory.
        
        Returns:
            None
        
        """

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.ok.return_value = True
        mock_session.chat_postMessage.return_value = mock_response
        mock_factory.return_value = mock_session

        message = "Test Slack alert message"
        channel = "#test-channel"
        result = send_slack_alert(message=message, 
                                  channel="#test-channel",
                                  oauth_token="xoxb-test-token",
                                  slack_bot="test-bot")

        assert result['status'] == "success"
        assert result['message'] == f"Alert sent to Slack: {channel} successfully"
    
 
class TestWriteFlaggedEventToSnowflake:
    """
    pass
    
    """
    @pytest.fixture(autouse=True)
    def snowflake_config_required_keys(self) -> dict:
        self.required_keys = [
            "snowflake_username",
            "snowflake_password",
            "snowflake_account",
            "snowflake_database",
            "snowflake_schema",
            "snowflake_warehouse",
            "snowflake_role",
        ]

class TestInputValidation(TestWriteFlaggedEventToSnowflake):

    @pytest.fixture(autouse=True)
    def non_dict_invalid_arg_types(self) -> list:
        self.non_dict_invalid_arg_types = [
            123,
            pd.Series([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            None,
            [],
            True,
            False,
            12.34,
            (1, 2, 3) 
        ]
        return self.non_dict_invalid_arg_types


    def test_event_not_dict_raises(self, 
                valid_snowflake_config: dict):
        with pytest.raises(Exception) as exc_info:
            for invalid_arg_type in self.non_dict_invalid_arg_types:
                write_flagged_event_to_snowflake(invalid_arg_type, valid_snowflake_config)
        
            assert exc_info.value.args[0]['status'] == 'error'
            assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
            assert exc_info.value.args[0]['error'] == 'event argument must be a dictionary'


    def test_snowflake_config_not_dict_raises(self):
        with pytest.raises(Exception) as exc_info:
            for invalid_arg_type in self.non_dict_invalid_arg_types:
                write_flagged_event_to_snowflake(valid_event, invalid_arg_type)
            assert exc_info.value.args[0]['status'] == 'error'
            assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
            assert exc_info.value.args[0]['error'] == 'snowflake_config argument must be a dictionary'

    def test_both_none_raises(self):
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(None, None)
        assert exc_info.value.args[0]['status'] == 'error'
        assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
        assert exc_info.value.args[0]['error'] == 'event argument must be a dictionary'

    def test_empty_dicts_pass_type_validation(self):
        """
        Empty dicts ARE dicts — type guards pass.
        KeyError on missing config keys is a separate, downstream concern.
        """
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake({}, {})

        assert exc_info.value.args[0]['status'] == 'error'
        assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
        assert exc_info.value.args[0]['error'] == f'missing required snowflake config key: {self.required_keys[0]}'


class TestSnowflakeConfigKeys(TestWriteFlaggedEventToSnowflake):
    """
    Each required config key, when missing, must raise — not silently
    produce a malformed connection string.
    """
    
    def test_missing_snowflake_config_key(self) -> None: 

        @pytest.mark.parametrize("missing_key", self.required_keys)
        def test_missing_config_key_raises(self, missing_key, valid_event, valid_snowflake_config):
            config = valid_snowflake_config.copy()
            del config[missing_key]
            with pytest.raises(Exception) as exc_info:
                write_flagged_event_to_snowflake(valid_event, config)
            assert exc_info.value.args[0]['status'] == 'error'
            assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
            assert exc_info.value.args[0]['error'] == f'missing required snowflake config key: {missing_key}'


class TestEventPayloadKeys(TestWriteFlaggedEventToSnowflake):
    """
    Each field bound in the INSERT must exist in the event dict.
    Missing keys must raise, not silently insert NULL.
    """

    REQUIRED_EVENT_KEYS = [
        "id", "ride_id", "flag_type", "rideable_type",
        "started_at", "ended_at",
        "start_station_name", "start_station_id",
        "end_station_name", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng",
        "member_casual",
    ]

    @pytest.mark.parametrize("missing_key", REQUIRED_EVENT_KEYS)
    def test_missing_event_key_raises(self, 
                                      missing_key,
                                      valid_event: dict, 
                                      valid_snowflake_config):
        event = valid_event.copy()
        del event[missing_key]
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(event, valid_snowflake_config)
        
        assert exc_info.value.args[0]['status'] == 'error'
        assert exc_info.value.args[0]['message'] == 'An error occurred while writing flagged event to Snowflake'
        assert exc_info.value.args[0]['error'] == f'missing required event key: {missing_key}'


class TestHappyPath:

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_returns_success_status(self, 
        mock_alerts_log: MagicMock,
        mock_session: MagicMock,
        valid_event: MagicMock,
        valid_snowflake_config: MagicMock
        ) -> None:
        
        _setup_mock_alerts_log(mock_alerts_log)
        _setup_mock_session(mock_session)
        
        result = write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        assert result["status"] == "success"
        assert result["message"] == "Flagged event written to Snowflake successfully"


    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_db_commit_is_called(self, 
                                 mock_alerts_log, 
                                 mock_session,
                                 valid_event: dict,
                                 valid_snowflake_config: dict) -> None:
        """Commit must fire — without it the INSERT is silently rolled back."""
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, mock_ctx = _setup_mock_session(mock_session)

        write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        mock_db.commit.assert_called_once()

        @patch("src.utils.create_session")
        @patch("src.utils.AlertsLog")
        def test_db_execute_called_once(self, mock_alerts_log, mock_session):
            """Guard against accidental double-inserts."""
            _setup_mock_alerts_log(mock_alerts_log)
            mock_db, _ = _setup_mock_session(mock_session)

            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
            assert mock_db.execute.call_count == 1

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_all_event_fields_passed_to_execute(self, mock_alerts_log, 
                                                mock_session,
                                                valid_event: dict,
                                                valid_snowflake_config: dict) -> None:
        """
        Every bound parameter in the INSERT must be sourced from the event.
        Verifies no field is accidentally hardcoded or silently dropped.
        """
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, _ = _setup_mock_session(mock_session)

        write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        _, call_kwargs = mock_db.execute.call_args
        # second positional arg is the params dict
        params = mock_db.execute.call_args[0][1]

        for key in [
            "id", "ride_id", "flag_type", "rideable_type",
            "started_at", "ended_at",
            "start_station_name", "start_station_id",
            "end_station_name", "end_station_id",
            "start_lat", "start_lng", "end_lat", "end_lng",
            "member_casual",
        ]:
            assert key in params, f"Missing param: {key}"
            assert params[key] == valid_event[key], (
                f"Param mismatch for {key}: "
                f"expected {valid_event[key]}, got {params[key]}"
            )


class TestConnectionString:

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_connection_string_contains_all_config_values(
        self, mock_alerts_log, mock_session,
          valid_event: dict,
            valid_snowflake_config: dict
    ):
        """
        Verifies the conn_string passed to create_session embeds every
        config value — a malformed string causes a silent auth failure.
        """
        _setup_mock_alerts_log(mock_alerts_log)
        _setup_mock_session(mock_session)

        write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        conn_string_used = mock_session.call_args[0][0]
        cfg = valid_snowflake_config

        assert cfg["snowflake_username"]  in conn_string_used
        assert cfg["snowflake_account"]   in conn_string_used
        assert cfg["snowflake_database"]  in conn_string_used
        assert cfg["snowflake_schema"]    in conn_string_used
        assert cfg["snowflake_warehouse"] in conn_string_used
        assert cfg["snowflake_role"]      in conn_string_used

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_connection_string_uses_snowflake_scheme(
        self, 
        mock_alerts_log,
          mock_session,
            valid_event: dict,
                valid_snowflake_config: dict
    ) -> None:
        _setup_mock_alerts_log(mock_alerts_log)
        _setup_mock_session(mock_session)

        write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        conn_string_used = mock_session.call_args[0][0]
        assert conn_string_used.startswith("snowflake://")

class TestDatabaseErrors:

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_operational_error_raises(self, 
                                      mock_alerts_log,
                                        mock_session,
                                        valid_event: dict,
                                        valid_snowflake_config: dict):
        
        """Network drop / warehouse suspended mid-query."""
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, mock_ctx = _setup_mock_session(mock_session)
        err_msg = "Snowflake connection lost"
        mock_db.execute.side_effect = OperationalError(
            err_msg, None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while writing flagged event to Snowflake"
        assert err_msg in exc_info.value.args[0]['error']

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_programming_error_raises(self, 
                        mock_alerts_log, mock_session,
                        valid_event: dict,
                        valid_snowflake_config: dict):
        """Schema mismatch — column doesn't exist, wrong type, etc."""
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, _ = _setup_mock_session(mock_session)
        err_msg = "column does not exist"
        mock_db.execute.side_effect = ProgrammingError(
            err_msg, None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while writing flagged event to Snowflake"
        assert err_msg in exc_info.value.args[0]['error']

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_interface_error_on_connect_raises(self, 
                                               mock_alerts_log,
                                                 mock_session,
                                                 valid_event: dict,
                                                 valid_snowflake_config: dict):
        """Auth failure — bad credentials, expired token."""
        _setup_mock_alerts_log(mock_alerts_log)
        err_msg = "Authentication failed"
        mock_session.side_effect = InterfaceError(
            err_msg, None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while writing flagged event to Snowflake"
        assert err_msg in exc_info.value.args[0]['error']

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_commit_failure_raises(self, 
                                   mock_alerts_log,
                                     mock_session,
                                     valid_event: dict,
                                     valid_snowflake_config: dict):
        """Execute succeeds but commit fails — partial write must not return success."""
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, _ = _setup_mock_session(mock_session)
        err_msg = "Connection lost during commit"
        mock_db.commit.side_effect = OperationalError(
            err_msg, None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while writing flagged event to Snowflake"
        assert err_msg in exc_info.value.args[0]['error']


    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_error_response_shape(self,
                                   mock_alerts_log,
                                     mock_session,
                                     valid_event: dict,
                                     valid_snowflake_config: dict):
        """Raised exception must carry the standard error envelope."""
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, _ = _setup_mock_session(mock_session)
        err_msg = "Connection lost"
        mock_db.execute.side_effect = OperationalError(
            err_msg, None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while writing flagged event to Snowflake"
        assert err_msg in exc_info.value.args[0]['error'] 

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_error_message_references_snowflake(self, 
                                                mock_alerts_log,
                                                  mock_session,
                                                  valid_event: dict,
                                                  valid_snowflake_config: dict):
        _setup_mock_alerts_log(mock_alerts_log)
        mock_db, _ = _setup_mock_session(mock_session)
        mock_db.execute.side_effect = OperationalError(
            "Connection lost", None, None
        )
        with pytest.raises(Exception) as exc_info:
            write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

        raised = ast.literal_eval(str(exc_info.value))
        assert "snowflake" in raised["message"].lower()


class TestSessionLifecycle:

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_session_used_as_context_manager(self,
     mock_alerts_log,
      mock_session,
      valid_event: dict, 
      valid_snowflake_config: dict):
        """create_session must be used as `with` — not called bare."""
        _setup_mock_alerts_log(mock_alerts_log)
        _setup_mock_session(mock_session)

        write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)
        
        mock_session.return_value.__enter__.assert_called_once()
        mock_session.return_value.__exit__.assert_called_once()

    @patch("src.utils.create_session")
    @patch("src.utils.AlertsLog")
    def test_session_exit_called_on_execute_failure(
            self, 
            mock_alerts_log,
             mock_session,
             valid_event: dict, 
             valid_snowflake_config: dict
        ):
            """
            Even when execute raises, the context manager __exit__ must fire.
            Ensures the session is not leaked on error.
            """
            _setup_mock_alerts_log(mock_alerts_log)
            mock_db, _ = _setup_mock_session(mock_session)
            mock_db.execute.side_effect = OperationalError(
                "Query failed", None, None
            )

            with pytest.raises(Exception):
                write_flagged_event_to_snowflake(valid_event, valid_snowflake_config)

            mock_session.return_value.__exit__.assert_called_once()

