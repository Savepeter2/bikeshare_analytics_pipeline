import os
import sys
from typing import Tuple, Dict
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (validate_raw_data, impute_missing_station_ids,
                       extract_source_data,
                       raw_data_schema,
                        validate_processed_data, clean_raw_data, InvalidArgumentTypeError)
from configs.logger_config import logger, error_logger
from configs.config import (RAW_S3_KEY, RAW_DATA_PATH, S3_RAW_BUCKET,
                            # LAST_ROW_INDEX, BATCH_SIZE,
                             SOURCE_BUCKET, SOURCE_S3_KEY,
                             AWS_ACCESS_KEY, AWS_SECRET_KEY, 
                            TRANSFORMED_BUCKET, 
                            CHUNK_SIZE, 
                             S3_CONFIG, SNOWFLAKE_CONFIG)
from src.models import create_session
import pandas as pd
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.models import BikeRide
from typing import List, Dict, Tuple, Set
from sqlalchemy import text
import numpy as np
import io
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta



def extract_and_validate_source_data(
        aws_access_key: str,
        aws_secret_access: str,
        source_bucket: str,
        source_s3_key: str,
        raw_bucket: str,
        raw_bucket_weekly_dump_prefix: str,
        batch_year: str,
        batch_week: str,
        chunk_size: int,
        raw_data_schema: Dict[str, str]
) -> Dict:
    
    """
    This function extracts the raw data from the given path and validates it

    Args:
        aws_access_key (str): AWS access key
        aws_secret_access (str): AWS secret access key
        source_bucket (str): Source s3 bucket name
        source_s3_key (str): Source s3 key
        raw_bucket_dump_prefix (str): Raw bucket dump prefix, folder in the raw bucket where extracted weekly trips will first be stored
        batch_year (str): Year of the trips to extract
        batch_week (str): Week of the trips to extract
    Returns:
        Dict: A dictionary containing the status, message, and raw data as a pandas DataFrame
    """
    try:
        if not all(isinstance(param, str) for param in [aws_access_key,
                                                            aws_secret_access, 
                                                            source_bucket,
                                                            source_s3_key,
                                                            ]):
            raise InvalidArgumentTypeError("All AWS parameters must be of type string")

        if not isinstance(batch_year, str):
            raise InvalidArgumentTypeError("batch_year must be a string")
        if not isinstance(batch_week, str):
            raise InvalidArgumentTypeError("batch_week must be a string")
        if not isinstance(chunk_size, int):
            raise InvalidArgumentTypeError("chunk_size must be an integer")
        if not isinstance(raw_data_schema, dict):
            raise InvalidArgumentTypeError("raw_data_schema must be a dictionary")
        
        extract_source_data_status = extract_source_data(
            aws_access_key,
            aws_secret_access,
                source_bucket,
                source_s3_key,
                batch_year,
                batch_week,
                chunk_size,
                raw_data_schema
            )

        if extract_source_data_status['status'] != 'success':
            raise Exception({
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": extract_source_data_status
            })
        
        raw_df = extract_source_data_status['data']
        validate_status = validate_raw_data(raw_df)

        if validate_status['status'] != 'success':
            raise Exception({
                "status": "error",
                "message": "Raw data validation failed",
                "error": validate_status
            })
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access
        )
        extracted_dump_s3_key = f"{raw_bucket_weekly_dump_prefix}/trips_year_{batch_year}_week_{batch_week}.csv"
        s3_client.put_object(
            Bucket=raw_bucket,
            Key=extracted_dump_s3_key,
            Body=raw_df.to_csv(index=False)
        )

        logger.info({
            "status": "success",
            "message": "Successfully extracted and validated raw data"
        })

        return {
            "status": "success",
            "message": "Successfully extracted and validated raw data",
            "extracted_dump_s3_key": extracted_dump_s3_key,
            "batch_year": batch_year,
            "batch_week": batch_week
        }

    except Exception as e:
        try:
            f_error = ast.literal_eval(str(e))
        
        except SyntaxError as se:
            f_error = str(e)

        error_logger.error({
            "status": "error",
            "message": "An error occurred while extracting and validating raw data",
            "error": f_error
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while extracting and validating raw data",
            "error": f_error
        })

class LoadToRawS3:
    def __init__(self, s3_config: Dict) -> None:
        """
        This class contains operations that load the raw bikeshare data to the raw S3 bucket.

        It ensures deduplication based and partititions by year and week.

        Partitioning Strategy:
            s3://raw_bucket/trips/
            ├── year=2022/
            │   ├── week=50/
            │   │   └── trips_year_2022_week_50.csv

        """
        try:
            if not isinstance(s3_config, dict):
                raise InvalidArgumentTypeError("s3_config must be a dictionary")
            if not isinstance(s3_config.get('access_key'), str):
                raise InvalidArgumentTypeError("access_key must be a string")
            if not isinstance(s3_config.get('secret_key'), str):
                raise InvalidArgumentTypeError("secret_key must be a string")
            if not isinstance(s3_config.get('raw_bucket'), str):
                raise InvalidArgumentTypeError("raw_bucket must be a string")
            if not isinstance(s3_config.get('raw_bucket_folder'), str):
                raise InvalidArgumentTypeError("raw_bucket_folder must be a string")
            if not isinstance(s3_config.get('raw_bucket_metadata_prefix'), str):
                raise InvalidArgumentTypeError("raw_bucket_metadata_prefix must be a string")
            if not isinstance(s3_config.get('raw_bucket_weekly_dump_prefix'), str):
                raise InvalidArgumentTypeError("raw_bucket_weekly_dump_prefix must be a string")

            self.s3_config = s3_config
            self.s3_client = boto3.client('s3',
                                    aws_access_key_id = s3_config['access_key'],
                                    aws_secret_access_key = s3_config['secret_key'])
            
            self.raw_bucket = s3_config['raw_bucket']
            self.raw_bucket_folder = s3_config.get('raw_bucket_folder')
            self.raw_bucket_metadata_prefix = s3_config.get('raw_bucket_metadata_prefix')

        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while initializing LoadToRawS3",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while initializing LoadToRawS3",
                "error": formatted_error
            })
    
    def generate_partition_path(self,
                                year: str,
                                 week: str) -> str:
        """
        Generates a S3 partition path based on ride datetime
        
        Args:
            year: Year of the partition
            week: Week number of the partition
        Returns:
            S3 partition path        
        """
        try:
            
            if not isinstance(year, str):
                raise InvalidArgumentTypeError("year must be a string")
            if not isinstance(week, str):
                raise InvalidArgumentTypeError("week must be a string")
            
            return (
                f"{self.raw_bucket_folder}/"
                f"year={year}/"
                f"week={week}/"
                f"ingestion_ts={datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')}/"
            )
    
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while generating partition path",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while generating partition path",
                "error": formatted_error
            })
    
    def get_batch_filename(self, year: str, week: str) -> str:
        """
        This generates a filename for each batch being processed based on the year, week, and current timestamp
        
        Args:
            year: Year of the batch
            week: Week of the batch
        Returns:
            Generated batch filename
        """
        try:
            if not isinstance(year, str):
                raise InvalidArgumentTypeError("year must be a string")
            if not isinstance(week, str):
                raise InvalidArgumentTypeError("week must be a string")
    
            file_name = "trips.csv"
            return file_name
    
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while generating batch filename",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while generating batch filename",
                "error": formatted_error
            })
    
    def get_ride_ids_from_metadata(self, 
                                   partition_year: str,
                                   partition_week: str) -> Set[str]:
        """
        Get ride IDs from metadata storage in the raw bucket.
        Args:
            partition_year: Year of the partition to retrieve IDs for
            partition_week: Week of the partition to retrieve IDs for
        Returns:
            Set of ride IDs
        """
        try:
            if not isinstance(partition_year, str):
                raise InvalidArgumentTypeError("partition_year must be a string")
            if not isinstance(partition_week, str):
                raise InvalidArgumentTypeError("partition_week must be a string")
            
            metadata_key = f"{self.raw_bucket_metadata_prefix}/year={partition_year}/week={partition_week}/ride_ids.parquet"
            response = self.s3_client.get_object(Bucket=self.raw_bucket, Key=metadata_key)
            parquet_data = response['Body'].read()  
            parquet_buffer = io.BytesIO(parquet_data)
            df = pd.read_parquet(parquet_buffer)
            ride_ids = set(df['id'].tolist())

            logger.info(
                {
                    "status": "info",
                    "message": f"Retrieved {len(ride_ids)} ride IDs for partition_year: {partition_year}, partition_week: {partition_week}"
                }
            )
            return {
                "status": "success",
                "message": f"Successfully retrieved {len(ride_ids)} ride IDs for partition year: {partition_year}, partition week: {partition_week}",
                "ride_ids": ride_ids
            }
        
        except self.s3_client.exceptions.NoSuchKey as e:
            return {
            "status": "success",
            "message": f"No ride IDs found for partition year: {partition_year}, partition week: {partition_week}, returning empty set",
            "ride_ids": set()
        }

        except Exception as e:
            try:
                f_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                f_error = str(e)

            logger.error({
                "status": "error",
                "message": f"An error occured retrieving ride IDs for partition year: {partition_year}, partition week: {partition_week}",
                "error": f_error
            })
            raise Exception({
                "status": "error",
                "message": f"An error occured retrieving ride IDs for partition year: {partition_year}, partition week: {partition_week}",
                "error": f_error 
            })
    
    def update_ride_ids_metadata(self, 
                                partition_year: str, 
                                partition_week: str,
                                new_ride_ids: Set[str]) -> Dict:
        """
        Update raw metadata with the newly processed ride IDs
        
        Args:
            partition_year: Year of the partition to update
            partition_week: Week of the partition to update
            new_ride_ids: Set of new ride IDs to add
        
        Returns:
            Dict: Status and message of the update operation
        """
        try:
            if not isinstance(partition_year, str):
                raise InvalidArgumentTypeError("partition_year must be a string")
            if not isinstance(partition_week, str):
                raise InvalidArgumentTypeError("partition_week must be a string")
            if not isinstance(new_ride_ids, set):
                raise InvalidArgumentTypeError("new_ride_ids must be a set")
            
            get_existing_ids = self.get_ride_ids_from_metadata(partition_year, partition_week)

            if get_existing_ids['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": f"Failed to retrieve existing ride IDs for {partition_year}, {partition_week}",
                    "error": get_existing_ids
                })
                return {
                    "status": "error",
                    "message": f"Failed to retrieve existing ride IDs for {partition_year}, {partition_week}",
                    "error": get_existing_ids
                }
            
            existing_ids = get_existing_ids['ride_ids']
            all_ids = existing_ids.union(new_ride_ids)
            all_ids = list(all_ids)
            df = pd.DataFrame({"id": all_ids})
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
            parquet_data = parquet_buffer.getvalue()

            metadata_key = f"{self.raw_bucket_metadata_prefix}/year={partition_year}/week={partition_week}/ride_ids.parquet"

            self.s3_client.put_object(
                Bucket=self.raw_bucket,
                Key=metadata_key,
                Body=parquet_data,
                ContentType='application/parquet'
            )

            logger.info({
                "status": "success",
                "message": f"Updated ride IDs for {partition_year}, {partition_week}",
                "updated_ids_count": len(all_ids)
            })

            return {
                "status": "success",
                "message": f"Updated ride IDs for {partition_year}, {partition_week}",
                "updated_ids_count": len(all_ids)
            }
    
        except Exception as e:
            try:
                f_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                f_error = str(e)
            logger.error({
                "status": "error",
                "message": f"An error occurred while updating ride IDs for {partition_year}, {partition_week}",
                "error": f_error
            })
            raise Exception({
                "status": "error",
                "message": f"An error occurred while updating ride IDs for {partition_year}, {partition_week}",
                "error": f_error
            })
    
    def filter_duplicate_rides(
            self, 
            rides_df: pd.DataFrame,
            batch_year: str,
            batch_week: str
    ) -> Dict:
        """
        Filters out duplicate rides based on existing ride IDs in metadata.
        
        Args:
            rides_df: DataFrame containing the rides to be filtered
        Returns:
            Dict: Status, message, and filtered DataFrame
        """
        try:
            if not isinstance(rides_df, pd.DataFrame):
                raise InvalidArgumentTypeError("rides_df must be a pandas DataFrame")
            
            if not isinstance(batch_year, str):
                raise InvalidArgumentTypeError("batch_year must be a string")
            
            if not isinstance(batch_week, str):
                raise InvalidArgumentTypeError("batch_week must be a string")

            if not rides_df.empty:
                # rides_df['start_datetime'] = pd.to_datetime(rides_df['started_at'], errors='coerce')
                # rides_df['partition_year'] = rides_df['start_datetime'].dt.strftime('%Y')
                # rides_df['partition_week'] = rides_df['start_datetime'].dt.strftime('%U')
                new_data_list = []
                existing_ids_list = []

                get_existing_ids = self.get_ride_ids_from_metadata(batch_year, batch_week)

                if get_existing_ids['status'] == 'error':
                    logger.error({
                        "status": "error",
                        "message": f"Failed to retrieve existing ride IDs for {batch_year}, {batch_week}",
                        "error": get_existing_ids
                    })
                    return {
                        "status": "error",
                        "message": f"Failed to retrieve existing ride IDs for {batch_year}, {batch_week}",
                        "error": get_existing_ids
                    }
                
                existing_ids = get_existing_ids['ride_ids']

                rides_df['ride_id'] = rides_df['ride_id'].astype(str)
                new_rides = rides_df[~rides_df['ride_id'].isin(existing_ids)]
                
                if not new_rides.empty:
                    new_data_list.append(new_rides)
            
                if new_data_list:
                    filtered_df = pd.concat(new_data_list, ignore_index=True)
                    
                    logger.info({
                        "status": "success",
                        "message": f"Filtered {len(filtered_df)} new rides from {len(rides_df)} total rides",
                        "filtered_rides_shape": filtered_df.shape
                    })
                    return {
                        "status": "success",
                        "message": f"Filtered {len(filtered_df)} new rides from {len(rides_df)} total rides",
                        "filtered_rides": filtered_df,
                        "existing_ride_ids": list(set().union(*existing_ids_list))
                    }
                
                else:
                    result_df = pd.DataFrame(columns=rides_df.columns)

                    logger.info({
                        "status": "success",
                        "message": "No new rides found, all rides are duplicates"
                    })
                    return {
                        "status": "success",
                        "message": "No new rides found, all rides are duplicates",
                        "filtered_rides": result_df
                    }
            else:
                logger.info({
                    "status": "success",
                    "message": "Input rides_df dataframe is empty, hence deduplication will be skipped in the raw bucket",
                    "input_shape": rides_df.shape
                })
                return {
                    "status": "success",
                    "message": "Input rides_df dataframe is empty, hence deduplication will be skipped in the raw bucket",
                    "input_shape": rides_df.shape,
                    "filtered_rides": rides_df
                }
            
        except Exception as e:
            try:
                f_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                f_error = str(e)
            logger.error({
                "status": "error",
                "message": "An error occurred while filtering duplicate rides",
                "error": f_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while filtering duplicate rides",
                "error": f_error
            })
    
    def upload_to_raw_bucket(
            self,
            new_rides_df: str,
            partition_year: str,
            partition_week: str,
    ) -> Dict:
        """
        Uploads the given DataFrame to the raw S3 bucket at the specified partition path and filename.
        
        Args:
            partition_path: S3 partition path
            batch_filename: Filename for the batch
            rides_df: DataFrame containing the rides to upload
        Returns:
            Dict: Status and message of the upload operation

        """
        try:
            if not isinstance(new_rides_df, pd.DataFrame):
                raise InvalidArgumentTypeError("rides_df must be a pandas DataFrame")
            
            if new_rides_df.empty:
                logger.info({
                    "status": "info",
                    "message": "No new rides to upload, the DataFrame is empty"
                })
                return {
                    "status": "success",
                    "message": "No new rides to upload, the DataFrame is empty",
                    "uploaded_rides_shape": new_rides_df.shape
                }

            uploaded_paths = []
            partiton_path = self.generate_partition_path(partition_year, partition_week)
            batch_filename = self.get_batch_filename(partition_year, partition_week)
            file_key = partiton_path + batch_filename
            self.s3_client.put_object(
                Bucket=self.raw_bucket,
                Key=file_key,
                Body=new_rides_df.to_csv(index=False)
            )
            s3_path = f"s3://{self.raw_bucket}/{file_key}"
            uploaded_paths.append(s3_path)

            no_records = new_rides_df.shape[0]

            logger.info({
                "status": "success",
                "message": f"Successfully uploaded {no_records} new rides to raw bucket",
                "uploaded_records": no_records
            })
            return {
                "status": "success",
                "message": f"Successfully uploaded {no_records} new rides to raw bucket",
                "uploaded_records": no_records
            }
        
        except Exception as e:
            try:
                f_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                f_error = str(e)
            logger.error({
                "status": "error",
                "message": "An error occurred while uploading to raw bucket",
                "error": f_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while uploading to raw bucket",
                "error": f_error
            })
    
    def load_raw_data_with_partitioning(
        self, 
        extracted_dump_s3_key: str,
        batch_year: str,
        batch_week: str
    ) -> Dict:
        """
        Loads raw data to S3 with partitioning and deduplication.
        
        Args:
            extracted_dump_s3_key: S3 key of the extracted raw data dump
            batch_year: Year of the batch
            batch_week: Week of the batch

        Returns:
            Dict: Status and message of the load operation

        """
        try:
            if not isinstance(extracted_dump_s3_key, str):
                raise InvalidArgumentTypeError("extracted_dump_s3_key must be a string")
            
            response = self.s3_client.get_object(Bucket=self.raw_bucket, 
                                                Key=extracted_dump_s3_key)
            
            extracted_raw_data = response['Body']
            extracted_raw_df = pd.read_csv(extracted_raw_data)

            filter_status = self.filter_duplicate_rides(extracted_raw_df,
                                                        batch_year,
                                                    batch_week)

            if filter_status['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": "Failed to filter duplicate raw trips from raw bucket",
                    "error": filter_status
                })
                return {
                    "status": "error",
                    "error": filter_status
                }
            
            filtered_raw_trips_df = filter_status['filtered_rides']
            extracted_ride_ids = set(filtered_raw_trips_df['ride_id'].astype(str).tolist())

            if filtered_raw_trips_df.empty:
                logger.info({
                    "status": "info",
                    "message": f"No new rides to upload for year: {batch_year}, week: {batch_week}. All rides are duplicates or empty."
                })
                return {
                    "status": "success",
                    "message": f"No new rides to upload for year: {batch_year}, week: {batch_week}. All rides are duplicates or empty.",
                    "uploaded_records": 0
                }
                
            upload_status = self.upload_to_raw_bucket(
                filtered_raw_trips_df,
                batch_year,
                batch_week
            )

            if upload_status['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": "Failed to upload new rides to raw bucket",
                    "error": upload_status
                })
                raise Exception({
                    "status": "error",
                    "message": "Failed to upload new rides to raw bucket",
                    "error": upload_status
                }
            )
            
            update_ride_ids_status = self.update_ride_ids_metadata(
                batch_year,
                batch_week,
                extracted_ride_ids
            )

            if update_ride_ids_status['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": "Failed to update ride IDs to raw bucket metadata",
                    "error": update_ride_ids_status
                })
                raise Exception({
                    "status": "error",
                    "message": "Failed to update ride IDs to raw bucket metadata",
                    "error": update_ride_ids_status
                })
            
            logger.info({
                "status": "success",
                "message": f"Successfully loaded raw data with partitioning for year: {batch_year}, week: {batch_week}",
                "uploaded_records": upload_status['uploaded_records'],
                "updated_ride_ids_count": update_ride_ids_status['updated_ids_count']
            })

            return {
                "status": "success",
                "message": f"Successfully loaded raw data to raw bucket for batch year: {batch_year}, week: {batch_week}",
                "uploaded_records": upload_status['uploaded_records'],
                "updated_ride_ids_count": update_ride_ids_status['updated_ids_count']
            }
        except Exception as e:
            try:
                f_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                f_error = str(e)
            logger.error({
                "status": "error",
                "message": "An error occurred while loading raw data to raw bucket",
                "error": f_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while loading raw data to raw bucket",
                "error": f_error
            })



class LoadToTransformedS3:
    def __init__(self, s3_config: Dict):
        """
        This class contains operations that load the cleaned bikeshare data to the transformed S3 bucket.
        It ensures deduplication and partitions by user_type and week.
        
        Partitioning Strategy:
        s3://transformed-bucket/cleaned_bikeshare/
        ├── user_type=casual/
        │   ├── year=2022/
        │   │   ├── week=50/
        │   │   │   └── batch_001.parquet
        └── user_type=member/
            ├── year=2022/
            │   ├── week=50/
            │   │   └── batch_002.parquet
        """
        try:
            if not isinstance(s3_config, dict):
                raise InvalidArgumentTypeError("s3_config must be a dictionary")
            
            if not isinstance(s3_config.get('access_key'), str):
                raise InvalidArgumentTypeError("access_key must be a string")
            
            if not isinstance(s3_config.get('secret_key'), str):
                raise InvalidArgumentTypeError("secret_key must be a string")
            
            if not isinstance(s3_config.get('source_bucket'), str):
                raise InvalidArgumentTypeError("source_bucket must be a string")
            
            if not isinstance(s3_config.get('source_s3_key'), str):
                raise InvalidArgumentTypeError("source_s3_key must be a string")
            
            if not isinstance(s3_config.get('raw_bucket_folder'), str):
                raise InvalidArgumentTypeError("raw_bucket_folder must be a string")
            
            if not isinstance(s3_config.get('raw_bucket_metadata_prefix'), str):
                raise InvalidArgumentTypeError("raw_bucket_metadata_prefix must be a string")
            
            if not isinstance(s3_config.get('raw_bucket_weekly_dump_prefix'), str):
                raise InvalidArgumentTypeError("raw_bucket_weekly_dump_prefix must be a string")
            
            if not isinstance(s3_config.get('transformed_bucket'), str):
                raise InvalidArgumentTypeError("transformed_bucket must be a string")
            
            if not isinstance(s3_config.get('raw_bucket'), str):
                raise InvalidArgumentTypeError("raw_bucket must be a string")
            
            if not isinstance(s3_config.get('raw_s3_key'), str):
                raise InvalidArgumentTypeError("raw_s3_key must be a string")
            
            if not isinstance(s3_config.get('transformed_bucket_metadata_prefix'), str):
                raise InvalidArgumentTypeError("transformed_bucket_metadata_prefix must be a string")
        
            self.s3_config = s3_config
            self.s3_client = boto3.client('s3',
                                        aws_access_key_id = s3_config['access_key'],
                                        aws_secret_access_key = s3_config['secret_key'])
            
            self.transformed_bucket_metadata_prefix = s3_config.get('transformed_bucket_metadata_prefix')
            self.transformed_bucket = s3_config['transformed_bucket']
            self.raw_bucket = s3_config['raw_bucket']
            self.raw_s3_key = s3_config['raw_s3_key']
            self.duplicate_tracker_prefix = f"{self.transformed_bucket_metadata_prefix}/metadata/processed_bikeshare"

        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while initializing LoadToTransformedS3",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while initializing LoadToTransformedS3",
                "error": formatted_error
            })

    def generate_partition_path(self,
                                user_type: str,
                                 year: str,
                                 week: str) -> str:
        """
        Generates a S3 partition path based on ride datetime
        
        Args:
            user_type: Type of user (e.g., casual, member)
            year: Year of the partition
            week: Week number of the partition
        Returns:
            S3 partition path        
        """
        try:
            if not isinstance(year, str):
                raise InvalidArgumentTypeError("year must be a string")
            if not isinstance(week, str):
                raise InvalidArgumentTypeError("week must be a string")
            if not isinstance(user_type, str):
                raise InvalidArgumentTypeError("user_type must be a string")     
            user_type = user_type.lower()   
            
            return (
                "cleaned_bikeshare/"
                f"user_type={user_type}/"
                f"year={year}/"
                f"week={week}/"
            )
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while generating partition path",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while generating partition path",
                "error": formatted_error
            })
    
    def get_batch_filename(self, batch_id: str) -> str:
        """
        This generates a filename for each batch being processed based on the batch id and current timestamp
        
        Args:
            batch_id: Identifier for the batch
        Returns:
            Generated batch filename
        
        """
        try:
            if not isinstance(batch_id, str):
                raise InvalidArgumentTypeError("batch_id must be a string")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{batch_id}_{timestamp}.parquet"
            return file_name
        
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while generating batch filename",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while generating batch filename",
                "error": formatted_error
            })
        

    def get_processed_ride_ids_from_metadata(self, 
                                             partition_year: str) -> Set[str]:
        """
        Get processed ride IDs from metadata storage in the transformed bucket.
        
        Args:
            partition_year: Year of the partition to retrieve IDs for
            
        Returns:
            Set of processed ride IDs
        """
        try:
            if not isinstance(partition_year, str):
                raise InvalidArgumentTypeError("partition_year must be a string")
            
            metadata_key = f"{self.duplicate_tracker_prefix}/{partition_year}/processed_ids.parquet"
            response = self.s3_client.get_object(Bucket=self.transformed_bucket, Key=metadata_key)
            parquet_data = response['Body'].read()  
            parquet_buffer = io.BytesIO(parquet_data)
            df = pd.read_parquet(parquet_buffer)
            processed_ids = set(df['id'].tolist())
    
            logger.info(
                {
                    "status": "info",
                    "message": f"Retrieved {len(processed_ids)} processed ride IDs for partition_year: {partition_year}"
                }
            )
            return {
                "status": "success",
                "message": f"Successfully retrieved {len(processed_ids)} processed ride IDs for partition year: {partition_year}",
                "processed_ids": processed_ids
            }
        
        except self.s3_client.exceptions.NoSuchKey as e:
            return {
            "status": "success",
            "message": f"No processed ride IDs found for partition year: {partition_year}, returning empty set",
            "processed_ids": set()
        }

        except Exception as e:
            logger.error({
                "status": "error",
                "message": f"An error occured retrieving processed ride IDs for partition year: {partition_year}",
                "error": str(e)
            })
            raise Exception({
                "status": "error",
                "message": f"An error occured retrieving processed ride IDs for partition year: {partition_year}",
                "error": str(e)
            })

    def update_processed_ride_ids_metadata(self, 
                                            partition_year: str, 
                                            new_ride_ids: Set[str]) -> Dict:
        """
        Update transformed metadata with the newly processed ride IDs
        
        Args:
            partition_year: Year of the partition to update
            new_ride_ids: Set of new ride IDs to add
        
        Returns:
            Dict: Status and message of the update operation
        """
        try:
            if not isinstance(partition_year, str):
                raise InvalidArgumentTypeError("partition_year must be a string")
            if not isinstance(new_ride_ids, set):
                raise InvalidArgumentTypeError("new_ride_ids must be a set")
            
            get_existing_ids = self.get_processed_ride_ids_from_metadata(partition_year)

            if get_existing_ids['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": f"Failed to retrieve existing ride IDs for {partition_year}",
                    "error": get_existing_ids
                })
                return {
                    "status": "error",
                    "message": f"Failed to retrieve existing ride IDs for {partition_year}",
                    "error": get_existing_ids
                }
            
            existing_ids = get_existing_ids['processed_ids']
            all_ids = existing_ids.union(new_ride_ids)
            all_ids = list(all_ids)
            df = pd.DataFrame({"id": all_ids})
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
            parquet_data = parquet_buffer.getvalue()

            metadata_key = f"{self.duplicate_tracker_prefix}/{partition_year}/processed_ids.parquet"

            self.s3_client.put_object(
                Bucket=self.transformed_bucket,
                Key=metadata_key,
                Body=parquet_data,
                ContentType='application/parquet'
            )

            logger.info({
                "status": "success",
                "message": f"Updated processed ride IDs for {partition_year}",
                "updated_ids_count": len(all_ids)
            })
            return {
                "status": "success",
                "message": f"Successfully updated processed ride IDs metadata for partition year: {partition_year}",
                "updated_ids_count": len(all_ids)
            }
        
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            logger.error({
                "status": "error",
                "message": f"An error occured while updating processed ride IDs for partition year: {partition_year}",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": f"An error occured while updating processed ride IDs for partition year: {partition_year}",
                "error": formatted_error
            })

    
    def filter_duplicate_rides(self, rides_df: pd.DataFrame) -> Dict:
        """
        Filter out duplicate rides based on existing data in S3 partitions
        Args:
            rides_df: DataFrame with rides data
        Returns:

        """
        try:
            if not isinstance(rides_df, pd.DataFrame):
                raise InvalidArgumentTypeError("rides_df must be a pandas DataFrame")

            if not rides_df.empty:
                rides_df['start_datetime'] = pd.to_datetime(
                    rides_df['started_at'], 
                    format='%Y-%m-%d %H:%M:%S', 
                    errors='raise' 
                )

                rides_df['partition_year'] = rides_df['start_datetime'].dt.strftime('%Y')

                new_data_list = []
                existing_ids_list = []

                for partition_year, group in rides_df.groupby('partition_year'):
                    get_existing_ids = self.get_processed_ride_ids_from_metadata(partition_year)

                    if get_existing_ids['status'] == 'error':
                        logger.error({
                            "status": "error",
                            "message": f"Failed to retrieve existing ride IDs for {partition_year}",
                            "error": get_existing_ids
                        })
                        raise Exception({
                            "status": "error",
                            "message": f"Failed to retrieve existing ride IDs for {partition_year}",
                            "error": get_existing_ids
                        })
                    
                    
                    existing_ids = get_existing_ids['processed_ids']
                    existing_ids_list.append(existing_ids)

                

                    group['ride_id'] = group['ride_id'].astype(str)
                    new_rides = group[~group['ride_id'].isin(existing_ids)]

                    if not new_rides.empty:
                        new_data_list.append(new_rides)
                
                if new_data_list:
                    
                    result_df = pd.concat(new_data_list, ignore_index=True)
                    
                    logger.info({
                        "status": "success",
                        "message": f"Filtered {len(result_df)} new rides from {len(rides_df)} total rides",
                        "filtered_rides_shape": result_df.shape
                    })
                    return {
                        "status": "success",
                        "message": f"Filtered {len(result_df)} new rides from {len(rides_df)} total rides",
                        "filtered_rides": result_df,
                        "existing_ids":  [elem for item in existing_ids_list for elem in (item if isinstance(item, set) else [item])]
                    }
                
                else:

                    result_df = pd.DataFrame(columns=rides_df.columns)

                    logger.info({
                        "status": "success",
                        "message": "No new rides found, all rides are duplicates",
                        "filtered_rides": result_df,
                    })

                    return {
                        "status": "success",
                        "message": "No new rides found, all rides are duplicates",
                        "filtered_rides": result_df
                    }
            else:
                logger.info({
                    "status": "success",
                    "message": "Input rides data is empty, hence deduplication will be skipped in the transformed bucket",
                    "input_shape": rides_df.shape

                })
                return {
                    "status": "success",
                    "message": "Input rides data is empty, hence deduplication will be skipped in the transformed bucket",
                    "filtered_rides": rides_df
                } 
        
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occured while filtering duplicate rides data",
                "error": formatted_error
            })
            raise Exception(
                {
                    "status": "error", 
                    "message": "An error occurred while filtering duplicate rides data",
                    "error": formatted_error
                }
            )

    def upload_partitioned_data(self, 
                               new_rides_df: pd.DataFrame,
                               partition_column1: str = 'member_casual',
                               partition_column2: str = 'started_at'
                               ) -> Dict:
        """
        Upload data to S3 with partitioning and compression
        
        Args:
            df: DataFrame to upload
            partition_column: Column to use for partitioning
            
        Returns:
            List of S3 paths where data was uploaded
        """
        try:
            if not isinstance(new_rides_df, pd.DataFrame):
                raise InvalidArgumentTypeError("new_rides_df must be a pandas DataFrame")
            
            if not isinstance(partition_column1, str):
                raise InvalidArgumentTypeError("partition_column1 must be a string")

            if not isinstance(partition_column2, str):
                raise InvalidArgumentTypeError("partition_column2 must be a string")
        
            if new_rides_df.empty:
                logger.info(
                    {
                        "status": "success",
                        "message": "No data to upload to transformed bucket, transformed data is empty",
                        "uploaded_paths": []
                    }
                )
                return {
                    "status": "success",
                    "message": "No data to upload to transformed bucket, transformed data is empty",
                    "uploaded_paths": []
                }
            
            uploaded_paths = []

            schema = pa.schema([
            ("ride_id", pa.string()),
            ("rideable_type", pa.string()),
            ("started_at", pa.string()),
            ("ended_at", pa.string()),
            ("start_station_name", pa.string()),
            ("start_station_id", pa.string()),
            ("end_station_name", pa.string()),
            ("end_station_id", pa.string()),
            ("start_lat", pa.string()),
            ("start_lng", pa.string()),
            ("end_lat", pa.string()),
            ("end_lng", pa.string()),
            ("member_casual", pa.string()),
            ("partition_key", pa.string()),
            ("year_week", pa.string())
             ])
            
            new_rides_df[partition_column2] = pd.to_datetime(new_rides_df[partition_column2], format = "%Y-%m-%d %H:%M:%S")
            new_rides_df['ended_at'] = pd.to_datetime(new_rides_df['ended_at'], format = "%Y-%m-%d %H:%M:%S")

            new_rides_df['year_week'] = new_rides_df[partition_column2].apply(
                lambda x: f"{x.year}-{x.isocalendar().week:02d}")

            new_rides_df[partition_column2] = new_rides_df[partition_column2].dt.strftime('%Y-%m-%d %H:%M:%S')
            new_rides_df['ended_at'] = new_rides_df['ended_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            new_rides_df['partition_key'] = new_rides_df[partition_column1] + "_" + new_rides_df['year_week']

            print("new_rides_df", new_rides_df)
            no_records = 0
            for partition_key, group in new_rides_df.groupby('partition_key'):
                user_type, year_week = partition_key.split('_')
                year, week = year_week.split('-')
                partition_path = self.generate_partition_path(user_type, year, week)
                max_batch_size = 100
                batch_counter = 1
                no_records_partition = 0
                for i in range(0, len(group), max_batch_size):
                    batch_df = group.iloc[i:i + max_batch_size].copy()

                    batch_id = f"batch_{batch_counter:03d}"
                    filename = self.get_batch_filename(batch_id)
                    s3_key = partition_path + filename
   
                    table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
                    buf = io.BytesIO()
                    pq.write_table(
                        table,
                        buf,
                        compression='snappy',
                        row_group_size=100_000,
                        use_dictionary=True,
                        write_statistics=True
                    )

                    buf.seek(0) 

                    self.s3_client.put_object(
                        Bucket=self.transformed_bucket,
                        Key=s3_key,
                        Body= buf
                    )
                    s3_path = f"s3://{self.transformed_bucket}/{s3_key}"
                    uploaded_paths.append(s3_path)
                    no_records_partition += len(batch_df)
                    batch_counter += 1

                no_records += no_records_partition

            logger.info({
                "status": "success",
                "message": f"Uploaded {len(uploaded_paths)} partitions to S3",
                "uploaded_records": no_records,
                "no_partitions": len(uploaded_paths)
            })
                
            return {
                "status": "success",
                "message": f"Uploaded {len(uploaded_paths)} partitions to S3",
                "uploaded_records": no_records,
                "no_partitions": len(uploaded_paths)
            }
        
        except Exception as e:
            logger.error({
                "status": "error",
                "message": f"Failed to upload partitioned data to s3",
                "error": str(e)
            })
            raise Exception({
                "status": "error",
                "message": "Failed to upload partitioned data to S3",
                "error": str(e)
            })
        

    def process_raw_data(self,
                         extracted_dump_s3_key: str,
                         raw_data_schema: Dict,                            
                            date_time_now: datetime = None) -> Dict:
        """
        This function processes the raw data and returns a cleaned and processed data
        Args:
            data (pd.DataFrame): Raw data as a pandas DataFrame
            last_row_index (int): The last row index to process

        Returns:
            Dict: A dictionary containing the status, message, and cleaned data as a pandas DataFrame

        """

        try:
            if not isinstance(extracted_dump_s3_key, str):
                raise InvalidArgumentTypeError("extracted_dump_s3_key must be a string type")
            
            if not isinstance(raw_data_schema, dict):
                raise InvalidArgumentTypeError("raw_data_schema must be a dictionary")
            
            #this needs to be changed to extracting from the raw bucket

            extract_raw_data_dump = self.s3_client.get_object(
                Bucket=self.raw_bucket,
                Key=extracted_dump_s3_key
            )
            if extract_raw_data_dump:
                if extract_raw_data_dump['ResponseMetadata']['HTTPStatusCode'] == 200:
                    raw_data_body = extract_raw_data_dump['Body']
                    raw_data_df = pd.read_csv(raw_data_body)
                else:
                    raise Exception({
                        "status": "error",
                        "message": "Failed to retrieve raw data dump from S3",
                        "error": extract_raw_data_dump['ResponseMetadata']
                    })
            else:
                raise Exception({
                    "status": "error",
                    "message": "An error occurred while retrieving extracted raw data dump from raw bucket",
                    "error": "No data found in S3 for the given key"
                })
            
            clean_data_status = clean_raw_data(raw_data_df)
            if clean_data_status['status'] != 'success':
                raise Exception({
                    "status": "error",
                    "message": "An error occurred while cleaning raw data",
                    "error": clean_data_status['error']
                })
            
            cleaned_data = clean_data_status["data"]

            process_station_data_status = impute_missing_station_ids(cleaned_data, batch_size=100)
            if process_station_data_status['status'] != 'success':
                raise Exception({
                    "status": "error",
                    "message": "An error occurred while imputing missing station IDs",
                    "error": process_station_data_status['error']
                })
            processed_data = process_station_data_status['data']
            date_time_now = date_time_now or datetime.now()
            #save this processed data to a parquet format in S3
            s3_key = f"processed_data/processed_bikeshare_{date_time_now.strftime('%Y%m%d_%H%M%S')}.parquet"
            parquet_buffer = io.BytesIO()
            processed_data.to_parquet(parquet_buffer, index=False, engine='pyarrow')
            parquet_data = parquet_buffer.getvalue()
            self.s3_client.put_object(
                Bucket=self.s3_config['transformed_bucket'],
                Key=s3_key,
                Body=parquet_data,
                ContentType='application/parquet'
            )

            logger.info({
                "status": "success",
                "message": "Successfully processed raw data"
            })

            return {
                "status": "success",
                "message": "Successfully processed raw data",
                "processed_data_key": s3_key
            }

        except Exception as e:
            error_logger.error({
                "status": "error",
                "message": "An error occurred while processing raw data",
                "error": str(e)
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while processing raw data",
                "error": str(e)
            })


    def process_rides_with_partitioning(self, 
                                        processed_data_key: str) -> Dict:

        """
        Completes the loading to the transformed S3 bucket with partitioning and deduplication.
        It pulls the processed data from the transformed bucket, filters out duplicates, 
        and uploads the new rides with partitioning by user_type and week.
        
        Args:
            processed_data_key: key of the processed data in the transformed bucket
        
        Returns:
            Dict: Status and message of the operation
            
        """
        try:
            if not isinstance(processed_data_key, str):
                raise InvalidArgumentTypeError("processed data key must be a string type")

            processed_data_obj = self.s3_client.get_object(Bucket=self.s3_config['transformed_bucket'], Key=processed_data_key)
            # print("processed_data_obj", processed_data_obj)

            processed_df = pd.read_parquet(io.BytesIO(processed_data_obj['Body'].read()))

            # print("processed_df head", processed_df)

            start_time = datetime.now()

            # print("starting to load staging data to s3, start_time: ", start_time)
            
            filter_rides = self.filter_duplicate_rides(processed_df)      
                   
            if filter_rides['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": f"Failed to filter duplicate rides: {filter_rides['message']}"
                })
                raise Exception({"status": "error",
                                "error": filter_rides
                })
            
            new_rides_df = filter_rides['filtered_rides']

            # print("new_rides_df shape after filtering duplicates: ", new_rides_df.head())

            if new_rides_df.empty:
                logger.info({
                    "status": "info",
                    "message": "No new rides to upload after filtering duplicates"
                })
                return {
                    "status": "success",
                    "message": "No new rides to upload after filtering duplicates",
                    "uploaded_records": 0,
                    "no_partitions": 0
                }
            

            upload_data_to_s3 = self.upload_partitioned_data(new_rides_df)

            print("upload_data_to_s3", upload_data_to_s3)
            
            if upload_data_to_s3['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": f"Failed to upload partitioned data to S3: {upload_data_to_s3['message']}"

                })
                return upload_data_to_s3
            
            uploaded_records = upload_data_to_s3['uploaded_records']
            no_partitions = upload_data_to_s3['no_partitions']
    
            # print("finished loading to staging bucket, time taken: ", datetime.now() -  start_time)
            
            for partition_year, group in new_rides_df.groupby('partition_year'):
                ride_ids = set(group['ride_id'].astype(str))
                self.update_processed_ride_ids_metadata(partition_year, ride_ids)

            logger.info({
                "status": "info",
                "message": "Successfully processed rides and uploaded with partitioning",
                "uploaded_records": uploaded_records,
                "no_partitions": no_partitions
            })
            return {
                "status": "success",
                "message": "Successfully processed rides and uploaded with partitioning",
                "uploaded_records": uploaded_records,
                "no_partitions": no_partitions
            }

        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)

            error_logger.error({
                "status": "error",
                "message": "An error occurred while trying to deduplicate and load processed data to the transformed bucket",
                "error": formatted_error
            })

            raise Exception({
                "status": "error",
                "message": "An error occurred while trying to deduplicate and load processed data to the transformed bucket",
                "error": formatted_error
            })



class LoadTransformedDataToSnowflake:
    """
    Class to load data from the transformed S3 bucket to Snowflake using SQLAlchemy ORM
    Attributes:
        snowflake_engine: SQLAlchemy engine connected to Snowflake
        s3_config (Dict): Dictionary containing S3 configuration details
    Methods:
        load_data: Load data from S3 to Snowflake table
    """
    def __init__(self, 
                snowflake_config: Dict,
                ):
        """
        Initialize connection to Snowflake and S3

        Args:
            snowflake_config (Dict): Dictionary containing Snowflake connection details

        Returns:
            None
        """
        try:
            if not isinstance(snowflake_config, dict):
                raise InvalidArgumentTypeError("snowflake_config must be a dictionary")
            
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
            snowflake_config_keys = list(snowflake_config.keys())
            missing_keys = set(required_keys) - set(snowflake_config_keys)
            missing_keys = list(missing_keys)
            if missing_keys:
                missing_keys = sorted(missing_keys)
                raise InvalidArgumentTypeError(f"Missing required key in snowflake_config: {missing_keys}")
            
            if not all(isinstance(value, str) for value in snowflake_config.values()):
                raise InvalidArgumentTypeError("All values in snowflake_config must be strings")

            account = snowflake_config['snowflake_account']
            user = snowflake_config['snowflake_username']
            password = snowflake_config['snowflake_password']
            warehouse = snowflake_config['snowflake_warehouse']
            database = snowflake_config['snowflake_database']
            schema = snowflake_config['snowflake_schema']
            role = snowflake_config['snowflake_role']
            stage_name = snowflake_config['stage_name']

            self.account = account
            self.user = user
            self.password = password
            self.warehouse = warehouse
            self.database = database
            self.schema = schema
            self.role = role
            self.stage_name = stage_name


            self.connection_string = (
                    f"snowflake://{self.user}:{self.password}@{self.account}/{self.database}/{self.schema}?warehouse={self.warehouse}&role={self.role}"
                )
            
            self.engine = create_engine(self.connection_string)
            self.session = sessionmaker(autocommit=False,
                                            autoflush=False,
                                            bind = self.engine)()
   
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            
            error_logger.error({
                "status": "error",
                "message": "An error occurred while initializing LoadTransformedDataToSnowflake class",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while initializing LoadTransformedDataToSnowflake class",
                "error": formatted_error
            })

    
    def load_from_stage_to_table(
            self,
    ) -> Dict:
        """
        This function loads data from the Snowflake stage to the target table using COPY INTO command.

        Args:
            None

        Returns:
            Dict: A dictionary containing the status, message, and number of new records loaded
        """
        try:
        
            with create_session(self.connection_string) as db:

                sql_query = text(f"""
                        COPY INTO {self.database}.{BikeRide.__table_args__['schema']}.{BikeRide.__tablename__}
                        FROM (
                        SELECT 
                            $1:ride_id,
                            $1:rideable_type,
                            $1:started_at,
                            $1:ended_at,
                            $1:start_station_name,
                            $1:start_station_id,
                            $1:end_station_name,
                            $1:end_station_id,
                            $1:start_lat,
                            $1:start_lng,
                            $1:end_lat,
                            $1:end_lng,
                            $1:member_casual,
                            CURRENT_TIMESTAMP() AS created_at,
                            CURRENT_TIMESTAMP() AS updated_at
                            FROM
                            @{self.database}.{BikeRide.__table_args__['schema']}.{self.stage_name}
                            )
                            PATTERN = '.*parquet'
                            ON_ERROR = 'ABORT_STATEMENT'
                            PURGE = FALSE
                            FORCE = FALSE
                            FILE_FORMAT = (
                            TYPE = 'parquet')
                        """
                        )
                            
                sql_query = db.execute(sql_query)
                cluster_query = text(f"ALTER TABLE {self.database}.{BikeRide.__table_args__['schema']}.{BikeRide.__tablename__} CLUSTER BY (ride_id, member_casual, started_at, end_station_id, start_station_id, ended_at);")
                db.execute(cluster_query)
                db.commit()

                no_records_loaded = db.query(BikeRide).count()

                logger.info({
                    "status": "success",
                    "message": f"Successfully loaded data from stage: {self.stage_name} to table: {BikeRide.__tablename__}",
                    "no_of_new_records": no_records_loaded
                })
                
                return {
                    "status": "success",
                    "message": f"Successfully loaded data from stage: {self.stage_name} to table: {BikeRide.__tablename__}",
                    "no_of_new_records": no_records_loaded
                }
            
        except Exception as e:
            try:
                formatted_error = ast.literal_eval(str(e))
            except SyntaxError as se:
                formatted_error = str(e)
            error_logger.error({
                "status": "error",
                "message": "An error occurred while loading data from stage to table",
                "error": formatted_error
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while loading data from stage to table",
                "error": formatted_error
            })
 

# chunk_size = CHUNK_SIZE
# aws_access_key = S3_CONFIG['access_key']
# aws_secret_access = S3_CONFIG['secret_key']
# source_bucket = S3_CONFIG['source_bucket']
# raw_bucket = S3_CONFIG['raw_bucket']
# source_s3_key = S3_CONFIG['source_s3_key']
# raw_bucket_folder = S3_CONFIG['raw_bucket_folder']
# raw_bucket_metadata_prefix = S3_CONFIG['raw_bucket_metadata_prefix']
# raw_bucket_weekly_dump_prefix = S3_CONFIG['raw_bucket_weekly_dump_prefix']
# batch_year = '2022' #all rides are in the year december, 2022
# batch_week = '48' #week 48 for testing, first week of december

# extraction_status = extract_and_validate_source_data(
#             aws_access_key,
#             aws_secret_access,
#             source_bucket,
#             source_s3_key,
#             raw_bucket,
#             raw_bucket_weekly_dump_prefix, 
#             batch_year, #all rides are in the year december, 2022
#             batch_week, #week 48 for testing, first week of december
#             chunk_size, #chunk size 1000 for testing
#             raw_data_schema
#         )

# extracted_dump_s3_key = extraction_status['extracted_dump_s3_key']

# load_to_raw_buk_obj = LoadToRawS3(S3_CONFIG)
    
# load_to_raw_bucket_status = load_to_raw_buk_obj.load_raw_data_with_partitioning(
#         extracted_dump_s3_key,
#         batch_year,
#         batch_week
#     )

# process_raw_data_obj = LoadToTransformedS3(s3_config = S3_CONFIG
# )   

# process_raw_data_status = process_raw_data_obj.process_raw_data(
#     extracted_dump_s3_key=extracted_dump_s3_key,
#     raw_data_schema=raw_data_schema
# )

# processed_s3_key = process_raw_data_status['processed_data_key']


# processed_s3_key = str(processed_s3_key)
# validate_processed_data_status = validate_processed_data(S3_CONFIG,
#                                                             processed_s3_key)

# processed_s3_key = str(processed_s3_key)
# load_processed_data_to_s3 = LoadToTransformedS3(S3_CONFIG).process_rides_with_partitioning(processed_s3_key)

# load_to_snowflake = LoadTransformedDataToSnowflake(
#         SNOWFLAKE_CONFIG
#     )
# load_status = load_to_snowflake.load_from_stage_to_table()

# if load_status['status'] != 'success':
#     raise Exception({
#         "status": "error",
#         "message": "Failed to load data from transformed S3 to Snowflake",
#         "error": load_status['error']
#     })



















chunk_size = CHUNK_SIZE
aws_access_key = S3_CONFIG['access_key']
aws_secret_access = S3_CONFIG['secret_key']
source_bucket = S3_CONFIG['source_bucket']
raw_bucket = S3_CONFIG['raw_bucket']
source_s3_key = S3_CONFIG['source_s3_key']
raw_bucket_folder = S3_CONFIG['raw_bucket_folder']
raw_bucket_metadata_prefix = S3_CONFIG['raw_bucket_metadata_prefix']
raw_bucket_weekly_dump_prefix = S3_CONFIG['raw_bucket_weekly_dump_prefix']

def get_batch_week_task(**context) -> Dict:
    # execution_date = context["logical_date"]
    execution_date = context['dag'].__dict__['default_args']['start_date']

    days_to_monday = execution_date.weekday()
    execution_week_monday = execution_date - timedelta(days=days_to_monday)
    preceding_week_monday = execution_week_monday - timedelta(days=7)

    iso = preceding_week_monday.isocalendar()
    batch_week = iso.week
    batch_year = iso.year
    logger.info(f"get_batch_week_task, batch_year: {batch_year}, batch_week: {batch_week}")

    context["ti"].xcom_push(key="batch_week", value=batch_week)
    context["ti"].xcom_push(key="batch_year", value=batch_year)

def extract_and_validate_source_task(**context) -> None:
    ti = context["ti"]

    batch_week = ti.xcom_pull(key="batch_week", task_ids="get_batch_week")
    batch_year = ti.xcom_pull(key="batch_year", task_ids="get_batch_week")

    if not batch_week or not batch_year:
        raise Exception({
            "status": "error",
            "message": "batch_week or batch_year from 'get_batch_week' XCom is None"
        })

    batch_week = str(batch_week)
    batch_year = str(batch_year)

    logger.info(f"extract_and_validate_source_task, batch_year: {batch_year}, batch_week: {batch_week}")

    if not isinstance(batch_week, str) or not isinstance(batch_year, str):
        raise InvalidArgumentTypeError("batch_week or batch_year from XCom is not a string type")

    extraction_status = extract_and_validate_source_data(
        aws_access_key,
        aws_secret_access,
        source_bucket,
        source_s3_key,
        raw_bucket,
        raw_bucket_weekly_dump_prefix,
        batch_year,
        batch_week,
        chunk_size,
        raw_data_schema,
    )

    if extraction_status["status"] != "success":
        raise Exception({
            "status": "error",
            "message": "Source data extraction and validation failed",
            "error": extraction_status["error"],
        })
    
    extracted_dump_s3_key = extraction_status["extracted_dump_s3_key"]

    ti.xcom_push(key="extracted_dump_s3_key", value=extracted_dump_s3_key)
    logger.info({
        "status": "success",
        "message": "Source data extraction and validation succeeded",
        "extract_and_validate_status": extraction_status
    })

def load_data_to_raw_bucket_task(**context) -> None:
    ti = context['ti']
    extracted_dump_s3_key = ti.xcom_pull(key='extracted_dump_s3_key', task_ids='extract_and_validate_source_data')
    if not extracted_dump_s3_key:
        raise Exception({
            "status": "error",
            "message": "extracted_dump_s3_key from 'extract_and_validate_source_data' XCom is None"
        })
    
    extracted_dump_s3_key = str(extracted_dump_s3_key)
    if not isinstance(extracted_dump_s3_key, str):
        raise InvalidArgumentTypeError("extracted_dump_s3_key must be a string type")

    batch_week = ti.xcom_pull(key='batch_week', task_ids='get_batch_week')
    batch_year = ti.xcom_pull(key='batch_year', task_ids='get_batch_week')
    if batch_week is None or batch_year is None:
        raise Exception({
            "status": "error",
            "message": "batch_week or batch_year from 'get_batch_week' Com is None"
        })

    batch_week = str(batch_week)
    batch_year = str(batch_year)

    if not isinstance(batch_week, str) or not isinstance(batch_year, str):
        raise InvalidArgumentTypeError("batch_week or batch_year must be string types")

    logger.info(f"load_data_to_raw_bucket_task, batch_year: {batch_year}, batch_week: {batch_week}")
    logger.info(f"load_data_to_raw_bucket_task, extracted_dump_s3_key: {extracted_dump_s3_key}")
    
    load_to_raw_buk_obj = LoadToRawS3(S3_CONFIG)
    
    load_to_raw_buk_obj.load_raw_data_with_partitioning(
        extracted_dump_s3_key,
        batch_year,
        batch_week
    )

def process_raw_data_task(**context) -> None:
    ti = context['ti']
    extracted_dump_s3_key = ti.xcom_pull(key='extracted_dump_s3_key', task_ids='extract_and_validate_source_data')
    
    if not extracted_dump_s3_key:
        raise Exception({
            "status": "error",
            "message": "extracted_dump_s3_key from 'extract_and_validate_source_data' XCom is None"
        })
    
    extracted_dump_s3_key = str(extracted_dump_s3_key)
    if not isinstance(extracted_dump_s3_key, str):
        raise InvalidArgumentTypeError("extracted_dump_s3_key must be a string type")
    process_raw_data_obj = LoadToTransformedS3(s3_config = S3_CONFIG)   

    process_raw_data_status = process_raw_data_obj.process_raw_data(
        extracted_dump_s3_key=extracted_dump_s3_key,
        raw_data_schema=raw_data_schema
    )
    processed_s3_key = process_raw_data_status['processed_data_key']
    context['ti'].xcom_push(key='processed_s3_key', value=processed_s3_key)

def validate_processed_data_task(**context) -> None:
    ti = context['ti']
    processed_s3_key = ti.xcom_pull(key='processed_s3_key', task_ids='process_raw_data')
    
    if not processed_s3_key:
        raise Exception({
            "status": "error",
            "message": "processed_s3_key from 'process_raw_data' XCom is None"
        })
    processed_s3_key = str(processed_s3_key)
    if not isinstance(processed_s3_key, str):
        raise InvalidArgumentTypeError("processed_s3_key must be a string type")
    
    validate_processed_data_status = validate_processed_data(S3_CONFIG,
                                                              processed_s3_key)
    if validate_processed_data_status['status'] != 'success':
        raise Exception({
            "status": "error",
            "message": "Processed data validation failed",
            "error": validate_processed_data_status['error']
        })

def load_processed_data_to_s3_task(**context) -> None:
    
    ti = context['ti']
    processed_s3_key = ti.xcom_pull(key='processed_s3_key', task_ids='process_raw_data')
    if not processed_s3_key:
        raise Exception({
            "status": "error",
            "message": "processed_s3_key from 'process_raw_data' XCom is None"
        })
    
    processed_s3_key = str(processed_s3_key)
    if not isinstance(processed_s3_key, str):
        raise InvalidArgumentTypeError("processed_s3_key must be a string type")
    
    load_processed_data_to_s3 = LoadToTransformedS3(S3_CONFIG).process_rides_with_partitioning(processed_s3_key)

    if load_processed_data_to_s3['status'] != 'success':
        raise Exception({
            "status": "error",
            "message": "Failed to upload processed data to transformed bucket",
            "error": load_processed_data_to_s3['error']
        })
    
    logger.info({
        "status": "success",
        "message": "Successfully uploaded processed data to transformed bucket",
        "uploaded_records": load_processed_data_to_s3['uploaded_records'],
        "no_partitions": load_processed_data_to_s3['no_partitions']
    })

def load_processed_s3_data_to_snowflake_task(**context) -> None:
    load_to_snowflake = LoadTransformedDataToSnowflake(
        SNOWFLAKE_CONFIG
    )
    load_status = load_to_snowflake.load_from_stage_to_table()

    if load_status['status'] != 'success':
        raise Exception({
            "status": "error",
            "message": "Failed to load data from transformed S3 to Snowflake",
            "error": load_status['error']
        })