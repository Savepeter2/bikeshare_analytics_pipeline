import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from src.etl import (extract_and_validate_source_data,
                     load_raw_data_to_s3,
                        LoadToTransformedS3,
                        LoadTransformedDataToSnowflake)
from re import Match
from tests.test_utils import (moto_boto3_client, mock_boto3_client, aws_credentials, 
                                create_transformed_bucket, upload_transformed_data_to_s3)
from pytest import fixture
import pandas as pd
from moto import mock_aws
from typing import Dict
from unittest.mock import patch, MagicMock
from botocore.client import BaseClient
import numpy as np
import boto3
import re
from typing import Generator, Dict
from datetime import datetime


# @pytest.fixture(scope="class")
# def aws_credentials():
#     """
#     These are FAKE credentials used only for mocking.
#     They will never connect to real AWS.
#     """
#     return {
#         'aws_access_key': 'AKIAIOSFODNN7EXAMPLE',
#         'aws_secret_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
#         'raw_bucket': "test-bucket",
#         'raw_s3_key': "raw/raw_data.parquet",
#         'transformed_bucket': "transformed-bucket",
#         'transformed_s3_key': "transformed/staging_data.parquet",
#         'source_bucket': "source-bucket",
#         'source_s3_key': "source/source_data.csv"

#     }


# @pytest.fixture(scope="class")
# def moto_boto3_client(aws_credentials: dict) -> Generator[BaseClient, None, None]:
#     """
#     Creates a mocked S3 client using moto. The real boto3 client is intercepted by moto.

#     Args:
#         aws_credentials: Fixture providing fake AWS credentials.

#     Returns:
#         A mocked boto3 S3 client.

#     """
#     with mock_aws():
#         access_key = aws_credentials['aws_access_key']
#         secret_key = aws_credentials['aws_secret_key']
#         yield boto3.client('s3', 
#                            aws_access_key_id=access_key,
#                            aws_secret_access_key=secret_key)
        

# @pytest.fixture(scope="class")
# def mock_boto3_client(aws_credentials: dict) -> BaseClient:
#     """
#     This fixture creates a mock boto3 client that simulates AWS interactions.
#     It checks the provided AWS credentials against the expected mocked credentials used in creating the moto client.

#     Args:
#         aws_credentials: Fixture providing mocked AWS credentials.
    
#     Returns:
#         A mocked boto3 client function.
#     """

#     boto3_client = boto3.client

#     def fake_client(service_name, **kwargs): 
#         access_key = kwargs.get("aws_access_key_id")
#         secret_key = kwargs.get("aws_secret_access_key")

#         if access_key != aws_credentials['aws_access_key'] or \
#            secret_key != aws_credentials['aws_secret_key']:
#             raise Exception("The security token included in the request is invalid")

#         return boto3_client(
#             service_name,
#             aws_access_key_id=aws_credentials['aws_access_key'],
#             aws_secret_access_key=aws_credentials['aws_secret_key']
#         )

#     return fake_client


# @pytest.fixture(scope="class")
# def 

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
                "end_station_name": [np.nan, np.nan, np.nan],
                "end_station_id": [np.nan, np.nan, np.nan],
                "start_lat": [40.7128, 34.0522, 41.8781],
                "start_lng": [-74.0060, -118.2437, -87.6298],
                "end_lat": [np.nan, np.nan, np.nan],
                "end_lng": [np.nan, np.nan, np.nan],
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
def raw_data_schema() -> Dict:
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

class TestExtractAndValidateSourceData:
    def test_invalid_aws_params(self):
        invalid_aws_params = [
            (123, "valid_key", "valid_bucket", "valid_s3_key"),
            ("valid_access_key", 456, "valid_bucket", "valid_s3_key"),
            ("valid_access_key", "valid_key", 789, "valid_s3_key"),
            ("valid_access_key", "valid_key", "valid_bucket", 101112)
        ]
        
        valid_batch_size = 100 #int
        valid_last_row_index = 1000 #int
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict
        
        for aws_access_key, aws_secret_key, source_bucket, source_s3_key in invalid_aws_params:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=aws_access_key,
                    aws_secret_access=aws_secret_key,
                    source_bucket=source_bucket,
                    source_s3_key=source_s3_key,
                    batch_size=valid_batch_size,
                    last_row_index=valid_last_row_index,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
            assert exc_info.value.args[0]['error'] == "All AWS parameters must be of type string"

    def test_invalid_batch_size(self) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when batch_size is not an integer.

        It tests various invalid types for batch_size including string, float, None, list, and dict. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_batch_sizes = ["100",
                                 100.5,
                                 (),
                                   None,
                                     [], {}]
        
        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_last_row_index = 30 #int
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict

        for batch_size in invalid_batch_sizes:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    batch_size=batch_size,
                    last_row_index=valid_last_row_index,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
            assert exc_info.value.args[0]['error'] == "batch_size must be an integer"
    
    def test_invalid_raw_data_schema(self) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when raw_data_schema is not a dictionary.

        It tests various invalid types for raw_data_schema including string, integer, float, None, list. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_raw_data_schemas = ["{'col1': 'str'}",
                                 100,
                                 100.5,
                                 (),
                                   None,
                                     []]
        
        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_batch_size = 100 #int
        valid_last_row_index = 30 #int

        for raw_data_schema in invalid_raw_data_schemas:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    batch_size=valid_batch_size,
                    last_row_index=valid_last_row_index,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
            assert exc_info.value.args[0]['error'] == "raw_data_schema must be a dictionary"

    def test_invalid_last_row_index(self) -> None:
        """
        Test that extract_and_validate_source_data raises an exception when last_row_index is not an integer.

        It tests various invalid types for last_row_index including string, float, None, list, and dict. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_last_row_indices = ["1000",
                                 1000.5,
                                   None,
                                   (),
                                     [], {}]
        
        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_batch_size = 100 #int
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict

        for last_row_index in invalid_last_row_indices:
            with pytest.raises(Exception) as exc_info:
                extract_and_validate_source_data(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    batch_size=valid_batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
            assert exc_info.value.args[0]['error'] == "last_row_index must be an integer"
    
    @patch('src.etl.extract_source_data')
    def test_error_from_extract_source_data(self, mock_extract_source_data) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by extract_source_data.

        This test mocks the extract_source_data function to raise an exception and verifies that
        extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            mock_extract_source_data: A mock object for the extract_source_data function.

        Returns:
            None

        """

        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_batch_size = 100 #int
        valid_last_row_index = 50 #int
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict

        mock_extract_source_data.side_effect = Exception({
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": "mocked error"
            })
        
        with pytest.raises(Exception) as exc_info:
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                batch_size=valid_batch_size,
                last_row_index=valid_last_row_index,
                raw_data_schema=valid_raw_data_schema
            )
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
        assert exc_info.value.args[0]['error'] == {
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": "mocked error"
            }
    
    @patch('src.etl.extract_source_data')
    def test_error_from_validate_raw_data(self,
                                          mock_extract_source_data: MagicMock) -> None:
        """
        Test that extract_and_validate_source_data handles exceptions raised by validate_raw_data.

        This test provides invalid raw data that will cause validate_raw_data to raise an exception.
        It verifies that extract_and_validate_source_data catches the exception and raises the appropriate error message.

        Args:
            None

        Returns:
            None

        """
        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_batch_size = 100 #int
        valid_last_row_index = 50 #int
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict


        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {valid_source_bucket}, s3 key: {valid_source_s3_key}",
            "data": pd.DataFrame({
                "col1": ["valid_string", "another_string"],
                "col2": ["invalid_int", "also_invalid"]
            })
        }



        with pytest.raises(Exception) as exc_info:            
            extract_and_validate_source_data(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                batch_size=valid_batch_size,
                last_row_index=valid_last_row_index,
                raw_data_schema=valid_raw_data_schema
            )
        assert exc_info.type is Exception
        assert exc_info.value.args[0]['status'] == "error"
        assert exc_info.value.args[0]['message'] == "An error occurred while extracting and validating raw data"
        missing_cols = [
            "ride_id", "rideable_type", "started_at", "ended_at",
            "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
        ]
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
    def test_successful_extraction_and_validation(self,
                                                  mock_extract_source_data: MagicMock,
                                                  mock_validate_raw_data: MagicMock) -> None:
        """
        Test that extract_and_validate_source_data successfully extracts and validates data.

        This test provides valid parameters and verifies that the function returns a success status
        along with the expected data.

        Args:
            None

        Returns:
            None

        """

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": "Successfully extracted data from source s3 bucket",
            "data": pd.DataFrame({
                "ride_id": ["ride1", "ride2"],
                "rideable_type": ["type1", "type2"],
                "started_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
                "ended_at": ["2023-01-01 10:30:00", "2023-01-01 11:30:00"],
                "start_lat": [40.7128, 34.0522],
                "start_lng": [-74.0060, -118.2437],
                "end_lat": [40.7138, 34.0532],
                "end_lng": [-74.0050, -118.2427],
                "member_casual": ["member", "casual"]
            })
        }

        mock_validate_raw_data.return_value = {
            "status": "success",
            "message": "Raw data validation successful"
        }


        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_source_bucket = "valid_bucket"
        valid_source_s3_key = "valid_s3_key"
        valid_batch_size = 100 #int
        valid_last_row_index = 50 #int
        valid_raw_data_schema = {
            "ride_id": "str",
            "rideable_type": "str",
            "started_at": "str",
            "ended_at": "str",
            "start_lat": "float",
            "start_lng": "float",
            "end_lat": "float",
            "end_lng": "float",
            "member_casual": "str"
        } #dict

        # For this test, we will assume that the extract_source_data function works correctly
        # and returns a DataFrame that matches the schema.
        
        result = extract_and_validate_source_data(
            aws_access_key=valid_aws_access_key,
            aws_secret_access=valid_aws_secret_key,
            source_bucket=valid_source_bucket,
            source_s3_key=valid_source_s3_key,
            batch_size=valid_batch_size,
            last_row_index=valid_last_row_index,
            raw_data_schema=valid_raw_data_schema
        )

        expected_df = pd.DataFrame({
                "ride_id": ["ride1", "ride2"],
                "rideable_type": ["type1", "type2"],
                "started_at": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"],
                "ended_at": ["2023-01-01 10:30:00", "2023-01-01 11:30:00"],
                "start_lat": [40.7128, 34.0522],
                "start_lng": [-74.0060, -118.2437],
                "end_lat": [40.7138, 34.0532],
                "end_lng": [-74.0050, -118.2427],
                "member_casual": ["member", "casual"]
            })
        
        assert result['status'] == "success"
        assert result['message'] == "Successfully extracted and validated raw data"
        pd.testing.assert_frame_equal(
            result['data'],
            expected_df)

class TestLoadRawDataToS3:
    """
    Test suite for the load_raw_data_to_s3 function.
    """

    def test_invalid_aws_params(self):
        invalid_aws_params = [
            (123, "valid_secret_key", "valid_bucket", "valid_raw_s3_key", "valid_raw_source_bucket", "valid_source_s3_key"), #non-string access key
            ("valid_access_key", 456, "valid_bucket", "valid_raw_s3_key", "valid_raw_source_bucket", "valid_source_s3_key"), #non-string secret key
            ("valid_access_key", "valid_secret_key", 789, "valid_raw_s3_key", "valid_raw_source_bucket", "valid_source_s3_key"), #non-string raw bucket
            ("valid_access_key", "valid_secret_key", "valid_bucket", 101112, "valid_raw_source_bucket", "valid_source_s3_key"), #non-string raw s3 key
            ("valid_access_key", "valid_secret_key", "valid_bucket", "valid_raw_s3_key", 131415, "valid_source_s3_key"), #non-string raw source bucket
            ("valid_access_key", "valid_secret_key", "valid_bucket", "valid_raw_s3_key", "valid_raw_source_bucket", 161718), #non-string raw source s3 key

        ]
        
        valid_raw_data_schema = {"col1": "str", "col2": "int"} #dict
        
        for aws_access_key, aws_secret_key, raw_bucket, raw_s3_key, raw_source_bucket, raw_source_s3_key in invalid_aws_params:
            with pytest.raises(Exception) as exc_info:
                load_raw_data_to_s3(
                    aws_access_key=aws_access_key,
                    aws_secret_access=aws_secret_key,
                    raw_bucket=raw_bucket,
                    raw_s3_key=raw_s3_key,
                    source_bucket=raw_source_bucket,
                    source_s3_key=raw_source_s3_key,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while loading raw data to S3"
            assert exc_info.value.args[0]['error'] == "all AWS parameters must be of type string"
    
    def test_invalid_raw_data_schema(self) -> None:
        """
        Test that load_raw_data_to_s3 raises an exception when raw_data_schema is not a dictionary.

        It tests various invalid types for raw_data_schema including string, integer, float, None, list. 
        
        The test verifies that the appropriate exception is raised with the expected error message.

        Args:
            None

        Returns:
            None

        """
        invalid_raw_data_schemas = ["{'col1': 'str'}",
                                 100,
                                 100.5,
                                 (),
                                   None,
                                     []]
        
        valid_aws_access_key = "valid_access_key"
        valid_aws_secret_key = "valid_secret_key"
        valid_raw_bucket = "valid_bucket"
        valid_raw_s3_key = "valid_s3_key"
        valid_source_bucket = "valid_source_bucket"
        valid_source_s3_key = "valid_source_s3_key"

        for raw_data_schema in invalid_raw_data_schemas:
            with pytest.raises(Exception) as exc_info:
                load_raw_data_to_s3(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    raw_bucket=valid_raw_bucket,
                    raw_s3_key=valid_raw_s3_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while loading raw data to S3"
            assert exc_info.value.args[0]['error'] == "raw_data_schema must be a dictionary"
    
    def test_error_from_source_bucket(self,
                                      mock_boto3_client: BaseClient,
                                      create_source_bucket: BaseClient,
                                      aws_credentials: Dict) -> None:
        """
        Test that load_raw_data_to_s3 handles exceptions raised when source bucket has no data.
        The source bucket must have data for a successful load to raw bucket.

        This test provides a mock S3 client that simulates an empty source bucket.
        It verifies that load_raw_data_to_s3 catches the exception and raises the appropriate error message.

        Args:
            mock_boto3_client: A mock boto3 S3 client.
            moto_boto3_client: A mocked boto3 client created using moto.
            aws_credentials: Mock AWS credentials.
            
        Returns:
            None

        """

        valid_aws_access_key = aws_credentials['aws_access_key']
        valid_aws_secret_key = aws_credentials['aws_secret_key']
        valid_raw_bucket = aws_credentials['raw_bucket']
        valid_raw_s3_key = aws_credentials['raw_s3_key']
        valid_source_bucket = aws_credentials['source_bucket']
        valid_source_s3_key = aws_credentials['source_s3_key']
        valid_raw_data_schema = {
            "ride_id": "str",
            "rideable_type": "str",
            "started_at": "str",
            "ended_at": "str",
            "start_lat": "float",
            "start_lng": "float",
            "end_lat": "float",
            "end_lng": "float",
            "member_casual": "str"
        } 

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            with pytest.raises(Exception) as exc_info:
                load_raw_data_to_s3(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    raw_bucket=valid_raw_bucket,
                    raw_s3_key=valid_raw_s3_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_data_schema=valid_raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while loading raw data to S3"
            assert exc_info.value.args[0]['error'] == {
                "status": "error",
                "message": "No data found in the source bucket",
                "error": 'An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.'
            }
    

    def test_successful_load_to_raw_bucket(self,
                                         mock_boto3_client: None,
                                        create_source_bucket: None,
                                        upload_to_source_bucket: None,
                                        create_raw_bucket: None,
                                         aws_credentials: Dict) -> None:
        """
        Test that load_raw_data_to_s3 successfully loads data from source bucket to raw bucket.
        The mocked data in the source bucket is extracted and loaded into the raw bucket.

        Args:
            mock_boto3_client: A mocked boto3 S3 client.
            create_source_bucket: Fixture to create source bucket.
            upload_to_source_bucket: Fixture to upload sample data to source bucket.
            create_raw_bucket: Fixture to create raw bucket.
            aws_credentials: Mock AWS credentials.
        Returns:
            None
        """

        valid_aws_access_key = aws_credentials['aws_access_key']
        valid_aws_secret_key = aws_credentials['aws_secret_key']
        valid_raw_bucket = aws_credentials['raw_bucket']
        valid_raw_s3_key = aws_credentials['raw_s3_key']
        valid_source_bucket = aws_credentials['source_bucket']
        valid_source_s3_key = aws_credentials['source_s3_key']
        valid_raw_data_schema = {
            "ride_id": "str",
            "rideable_type": "str",
            "started_at": "str",
            "ended_at": "str",
            "start_lat": "float",
            "start_lng": "float",
            "end_lat": "float",
            "end_lng": "float",
            "member_casual": "str"
        }

        upload_to_source_bucket()

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            result = load_raw_data_to_s3(
                aws_access_key=valid_aws_access_key,
                aws_secret_access=valid_aws_secret_key,
                raw_bucket=valid_raw_bucket,
                raw_s3_key=valid_raw_s3_key,
                source_bucket=valid_source_bucket,
                source_s3_key=valid_source_s3_key,
                raw_data_schema=valid_raw_data_schema
            )

            assert result['status'] == "success"
            assert result['message'] == f"Successfully loaded raw data to raw s3 bucket: {valid_raw_bucket}, s3 key: {valid_raw_s3_key}"

    def test_existing_raw_data(self,
                                mock_boto3_client: BaseClient,
                                create_source_bucket: None,
                                upload_to_source_bucket: None,
                                create_raw_bucket: None,
                                aws_credentials: Dict) -> None:
            """
            Test that load_raw_data_to_s3 successfully skips loading when raw data already exists in the raw bucket.
            The data is a static data, so it should be uploaded only once.
    
            Args:
                mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
                upload_to_source_bucket: Fixture to upload sample data to source bucket.
                aws_credentials: Mock AWS credentials.
                
            Returns:
                None

            """
            valid_aws_access_key = aws_credentials['aws_access_key']
            valid_aws_secret_key = aws_credentials['aws_secret_key']
            valid_raw_bucket = aws_credentials['raw_bucket']
            valid_raw_s3_key = aws_credentials['raw_s3_key']
            valid_source_bucket = aws_credentials['source_bucket']
            valid_source_s3_key = aws_credentials['source_s3_key']
            valid_raw_data_schema = {
                "ride_id": "str",
                "rideable_type": "str",
                "started_at": "str",
                "ended_at": "str",
                "start_lat": "float",
                "start_lng": "float",
                "end_lat": "float",
                "end_lng": "float",
                "member_casual": "str"
            }

            upload_to_source_bucket()

            with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
                # First load to create the raw data
                load_raw_data_to_s3(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    raw_bucket=valid_raw_bucket,
                    raw_s3_key=valid_raw_s3_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_data_schema=valid_raw_data_schema
                )

                # # Second load should detect existing raw data and skip loading
                result = load_raw_data_to_s3(
                    aws_access_key=valid_aws_access_key,
                    aws_secret_access=valid_aws_secret_key,
                    raw_bucket=valid_raw_bucket,
                    raw_s3_key=valid_raw_s3_key,
                    source_bucket=valid_source_bucket,
                    source_s3_key=valid_source_s3_key,
                    raw_data_schema=valid_raw_data_schema
                )

                assert result['status'] == "success"
                assert result['message'] == f"Raw data already exists at s3://{valid_raw_bucket}/{valid_raw_s3_key}. Raw data load skipped."
    
    def test_unexpected_error(self,
                              mock_boto3_client: BaseClient,
                              create_source_bucket: None,
                              upload_to_source_bucket: None,
                              aws_credentials: Dict) -> None:
            """
            Test that load_raw_data_to_s3 handles unexpected exceptions.

            This test simulates an unexpected exception during the load process and verifies that
            load_raw_data_to_s3 catches the exception and raises the appropriate error message.

            Args:
                mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
                create_source_bucket: Fixture to create source bucket.
                upload_to_source_bucket: Fixture to upload sample data to source bucket.
                aws_credentials: Mock AWS credentials.

            Returns:
                None

            """
            valid_aws_access_key = aws_credentials['aws_access_key']
            valid_aws_secret_key = aws_credentials['aws_secret_key']
            valid_raw_bucket = aws_credentials['raw_bucket']
            valid_raw_s3_key = aws_credentials['raw_s3_key']
            valid_source_bucket = aws_credentials['source_bucket']
            valid_source_s3_key = aws_credentials['source_s3_key']
            valid_raw_data_schema = {
                "ride_id": "str",
                "rideable_type": "str",
                "started_at": "str",
                "ended_at": "str",
                "start_lat": "float",
                "start_lng": "float",
                "end_lat": "float",
                "end_lng": "float",
                "member_casual": "str"
            }

            upload_to_source_bucket()

            with patch('src.etl.boto3.client', side_effect=Exception("Unexpected error with boto3")):
                with pytest.raises(Exception) as exc_info:
                    load_raw_data_to_s3(
                        aws_access_key=valid_aws_access_key,
                        aws_secret_access=valid_aws_secret_key,
                        raw_bucket=valid_raw_bucket,
                        raw_s3_key=valid_raw_s3_key,
                        source_bucket=valid_source_bucket,
                        source_s3_key=valid_source_s3_key,
                        raw_data_schema=valid_raw_data_schema
                    )
        
                assert exc_info.type is Exception
                assert exc_info.value.args[0]['status'] == "error"
                assert exc_info.value.args[0]['message'] == "An error occurred while loading raw data to S3"
                assert exc_info.value.args[0]['error'] == "Unexpected error with boto3"

class TestLoadToTransformedS3:
    """
    This is a test suite for the LoadToTransformedS3 class methods.

    """

    def test_invalid_s3_config_arg(
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
        invalid_s3_configs = [
            "invalid_s3_config",
            123,
            45.67,
            (),
            None,
            []
        ]
     
        for s3_config in invalid_s3_configs:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config)
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            assert exc_info.value.args[0]['error'] == "s3_config must be a dictionary"
    
    def test_invalid_s3_config_keys(
        self,
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
        invalid_s3_configs = [
            (123, "valid_secret_key", "valid_transformed_bucket", "valid_raw_bucket", "valid_raw_s3_key", "valid_base_prefix"), #missing access_key
            ("valid_access_key", 456, "valid_transformed_bucket", "valid_raw_bucket", "valid_raw_s3_key", "valid_base_prefix"), #missing secret_access_key
            ("valid_access_key", "valid_secret_key", 789, "valid_raw_bucket", "valid_raw_s3_key", "valid_base_prefix"), #missing transformed_bucket
            ("valid_access_key", "valid_secret_key", "valid_transformed_bucket", 101112, "valid_raw_s3_key", "valid_base_prefix"), #missing raw_bucket
            ("valid_access_key", "valid_secret_key", "valid_transformed_bucket", "valid_raw_bucket", 131415, "valid_base_prefix"), #missing raw_s3_key
            ("valid_access_key", "valid_secret_key", "valid_transformed_bucket", "valid_raw_bucket", "valid_raw_s3_key", 161718), #missing base_prefix
        ]
        # s3_config = {
        #     "access_key": 123, #non-string access_key
        #     "secret_access_key": 111, #non-string secret_access_key
        #     "transformed_bucket": 000, #non-string transformed_bucket
        #     "raw_bucket": 456, #non-string raw_bucket
        #     "raw_s3_key": 500, #non-string raw_s3_key
        #     "base_prefix": 789.4 #non-string base_prefix
        # }
        # s3_config_headers = list(set(s3_config.keys()))
        # for key in s3_config_headers:
        for access_key, secret_key, transformed_bucket, raw_bucket, raw_s3_key, base_prefix in invalid_s3_configs:

            s3_config = {
                "access_key": access_key,
                "secret_key": secret_key,
                "transformed_bucket": transformed_bucket,
                "raw_bucket": raw_bucket,
                "raw_s3_key": raw_s3_key,
                "base_prefix": base_prefix
            }
            with pytest.raises(Exception) as exc_info:
                loadtos3obj = LoadToTransformedS3(s3_config=s3_config)
            print("s3_config", s3_config)
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadToTransformedS3"
            if not isinstance(access_key, str):
                assert exc_info.value.args[0]['error'] == f"access_key must be a string"
            elif not isinstance(secret_key, str):
                assert exc_info.value.args[0]['error'] == f"secret_key must be a string"
            elif not isinstance(transformed_bucket, str):
                assert exc_info.value.args[0]['error'] == f"transformed_bucket must be a string"
            elif not isinstance(raw_bucket, str):
                assert exc_info.value.args[0]['error'] == f"raw_bucket must be a string"
            elif not isinstance(raw_s3_key, str):
                assert exc_info.value.args[0]['error'] == f"raw_s3_key must be a string"
            elif not isinstance(base_prefix, str):
                assert exc_info.value.args[0]['error'] == f"base_prefix must be a string"

    def test_generate_partition_path_invalid_args(
        self,
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


        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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
        aws_credentials: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.batch_filename method generates the correct batch filename.

        This test provides a valid batch index and verifies that the generated filename matches the expected format.

        Args:
            aws_credentials: Mocked AWS credentials.
        Returns:
            None
        """
        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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
        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
        
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        create_transformed_bucket: None,
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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
        aws_credentials: Dict
    ) -> None:
        """
        Test that LoadToTransformedS3.upload_partitioned_data method successfully uploads partitioned data to the transformed S3 bucket.
        This test verifies that the method completes without raising exceptions when provided with valid inputs.

        Args:
            mock_boto3_client: A mocked boto3 client fixture used to patch boto3.client.
            create_transformed_bucket: Fixture to create transformed bucket.
            aws_credentials: Mocked AWS credentials.
        
        Returns:
            None
        """

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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
        
    def test_process_raw_data_invalid_batch_size(
        self,
        aws_credentials: Dict,
    ) -> None:
        """
        Test that ETLProcessor.process_raw_data method raises an exception when an invalid batch size is provided.
        The batch_size argument must be a positive integer.

        Args:
            aws_credentials: Mocked AWS credentials.

        Returns:
            None
        """
        invalid_batch_sizes = [
            -10, 
            0, 
            3.5, 
            "ten", 
            None,
            [],
            {}
        ] #batch size must be a positive integer
        last_row_index = 0
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
        for batch_size in invalid_batch_sizes:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj.process_raw_data(
                    batch_size=batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while processing raw data"
            assert exc_info.value.args[0]['error'] == "batch_size must be a positive integer"
        
    def test_process_raw_data_invalid_last_row_index(
        self,
        aws_credentials: Dict,
        raw_data_schema: Dict
    ) -> None:
        """
        Tests the the LoadToTransformedS3.process_raw_data method raises an exception when an invalid last_row_index is provided.
        The last_row_index argument must be a non-negative integer.
        This test verifies that the appropriate exception is raised with the expected error message.

        Args:
            aws_credentials: Mocked AWS credentials.
            raw_data_schema: Fixture providing the expected schema for the raw data.

        Returns:
            None
        """
        invalid_last_row_indices = [
            -5, 
            2.7, 
            "five", 
            None,
            [],
            {}
        ] #last_row_index must be a non-negative integer
        batch_size = 10

        

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
        for last_row_index in invalid_last_row_indices:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj.process_raw_data(
                    batch_size=batch_size,
                    last_row_index=last_row_index,
                    raw_data_schema=raw_data_schema
                )
        
            assert exc_info.type is Exception
            assert exc_info.value.args[0]['status'] == "error"
            assert exc_info.value.args[0]['message'] == "An error occurred while processing raw data"
            assert exc_info.value.args[0]['error'] == "last_row_index must be a non-negative integer"
    
    def test_process_raw_data_invalid_raw_data_schema(
        self,
        aws_credentials: Dict,
    ) -> None:
        """
        Tests the the LoadToTransformedS3.process_raw_data method raises an exception when an invalid raw_data_schema is provided.
        The raw_data_schema argument must be a dictionary.
        This test verifies that the appropriate exception is raised with the expected error message.

        Args:
            aws_credentials: Mocked AWS credentials.

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
        batch_size = 10
        last_row_index = 0

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
        for raw_data_schema in invalid_raw_data_schemas:
            with pytest.raises(Exception) as exc_info:
                loadtos3obj.process_raw_data(
                    batch_size=batch_size,
                    last_row_index=last_row_index,
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
        aws_credentials: Dict,
        raw_data_schema: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
        batch_size = 10
        last_row_index = 0

        source_bucket = aws_credentials['source_bucket']
        source_s3_key = aws_credentials['source_s3_key']
        start_index = last_row_index + 1
        end_index = last_row_index + batch_size - 1

        raw_df = pd.DataFrame({
                "ride_id": ["ride20", "ride21"],
                "rideable_type": ["Electric_bike", "Docked_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00"],
                "start_station_name": ["Start_station1", "start_station2"],
                "start_station_id": ["Start_Station_id1", "Start_sTation_id2"],
                "end_station_name": [np.nan, np.nan],
                "end_station_id": [np.nan, np.nan],
                "start_lat": ["41.8781", "41.8810"],
                "start_lng": ["-87.6298", "-87.6300"],
                "end_lat": ["41.8850", "41.8860"],
                "end_lng": ["-87.6340", "-87.6350"],
                "member_casual": ["casual", "member"]
            })
        
        cleaned_df = pd.DataFrame(
            {
                "ride_id": ["ride20", "ride21"],
                "rideable_type": ["electric_bike", "docked_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00"],
                "start_station_name": ["start_station1", "start_station2"],
                "start_station_id": ["start_station_id1", "start_station_id2"],
                "end_station_name": [np.nan, np.nan],
                "end_station_id": [np.nan, np.nan],
                "start_lat": ["41.8781", "41.8810"],
                "start_lng": ["-87.6298", "-87.6300"],
                "end_lat": ["41.8850", "41.8860"],
                "end_lng": ["-87.6340", "-87.6350"],
                "member_casual": ["casual", "member"]
            }
        )

        imputed_df = pd.DataFrame(
            {
                "ride_id": ["ride20", "ride21"],
                "rideable_type": ["electric_bike", "docked_bike"],
                "started_at": ["2023-04-03 08:00:00", "2023-04-03 09:00:00"],
                "ended_at": ["2023-04-03 08:30:00", "2023-04-03 09:30:00"],
                "start_station_name": ["start_station1", "start_station2"],
                "start_station_id": ["start_station_id1", "start_station_id2"],
                "end_station_name": ["end_station1", "end_station2"],
                "end_station_id": ["end_station_id1", "end_station_id2"],
                "start_lat": ["41.8781", "41.8810"],
                "start_lng": ["-87.6298", "-87.6300"],
                "end_lat": ["41.8850", "41.8860"],
                "end_lng": ["-87.6340", "-87.6350"],
                "member_casual": ["casual", "member"]
            }
        )

        upload_to_source_bucket()

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}, rows: {start_index} to {end_index}",
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

        date_time_now = datetime(2025, 12, 30, 12, 0, 0)
        expected_s3_key = f"processed_data/processed_bikeshare_{date_time_now.strftime('%Y%m%d_%H%M%S')}.parquet"

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_result = loadtos3obj.process_raw_data(
                batch_size=batch_size,
                last_row_index=last_row_index,
                raw_data_schema=raw_data_schema,
                date_time_now=date_time_now
            )
            assert process_result['status'] == "success"
            assert process_result['message'] == f"Successfully processed raw data"
            assert process_result['processed_data_key'] == expected_s3_key

    def test_process_rides_with_partitioning_invalid_args(
        self,
        aws_credentials: Dict,
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
            'access_key': aws_credentials['aws_access_key'],
            'base_prefix': aws_credentials['transformed_bucket']
        }

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)
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
        aws_credentials: Dict
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

        valid_s3_config = {
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }
     
        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

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
        raw_data_schema: Dict
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
        valid_s3_config = {
            'source_bucket': aws_credentials['source_bucket'],
            'source_s3_key': aws_credentials['source_s3_key'],
           'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key'],
        }

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

        batch_size = 10
        last_row_index = 0
    
        source_bucket = valid_s3_config['source_bucket']
        source_s3_key = valid_s3_config['source_s3_key']
        start_index = last_row_index + 1
        end_index = last_row_index + batch_size - 1

        upload_sample_to_source_bucket = upload_to_source_bucket(raw_df)

        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}, rows: {start_index} to {end_index}",
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

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_raw_data = loadtos3obj.process_raw_data(
                batch_size=10,
                last_row_index=0,
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
        raw_data_schema: Dict
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
        valid_s3_config = {
            'source_bucket': aws_credentials['source_bucket'],
            'source_s3_key': aws_credentials['source_s3_key'],
            'transformed_bucket': aws_credentials['transformed_bucket'],
            'raw_bucket': aws_credentials['raw_bucket'],
            'access_key': aws_credentials['aws_access_key'],
            'secret_key': aws_credentials['aws_secret_key'],
            'base_prefix': aws_credentials['transformed_bucket'],
            'raw_s3_key': aws_credentials['raw_s3_key']
        }

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
        batch_size = 5
        last_row_index = 0

        source_bucket = valid_s3_config['source_bucket']
        source_s3_key = valid_s3_config['source_s3_key']
        start_index = last_row_index + 1
        end_index = last_row_index + batch_size - 1

        upload_sample_to_source_bucket = upload_to_source_bucket(test_duplicate_df)
        mock_extract_source_data.return_value = {
            "status": "success",
            "message": f"Successfully extracted data from source s3 bucket: {source_bucket}, s3 key: {source_s3_key}, rows: {start_index} to {end_index}",
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

        loadtos3obj = LoadToTransformedS3(s3_config=valid_s3_config)

        with patch('src.etl.boto3.client', side_effect=mock_boto3_client):
            process_raw_data = loadtos3obj.process_raw_data(
                batch_size=batch_size,
                last_row_index=last_row_index,
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

    # def test_invalid_instantiation_args(self) -> None:
        
    #     """
    #     This test verifies that the LoadTransformedDataToSnowflake class raises the appropriate exceptions when instantiated with invalid arguments.
    #     The test checks for invalid types for the s3_config and snowflake_config parameters.

    #     Args:
    #         None

    #     Returns:
    #         None
    #     """
    #     invalid_snowflake_config = [
    #         "invalid_config", 
    #         123, 
    #         45.67, 
    #         None,
    #         [],
    #         set()
    #     ] #snowflake_config must be a dictionary

    #     for snowflake_config in invalid_snowflake_config:
    #         with pytest.raises(Exception) as exc_info:
    #             LoadTransformedDataToSnowflake(
    #                 snowflake_config=snowflake_config
    #             )
        
    #         assert exc_info.type is Exception
    #         assert exc_info.value.args[0]['status'] == "error"
    #         assert exc_info.value.args[0]['message'] == "An error occurred while initializing LoadTransformedDataToSnowflake class"
    #         assert exc_info.value.args[0]['error'] == "snowflake_config must be a dictionary"
    
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
