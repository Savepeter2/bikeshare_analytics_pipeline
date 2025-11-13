import os
import sys
from typing import Tuple, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (validate_raw_data, impute_missing_station_ids,
                       extract_source_data,
                       raw_data_schema,
                        validate_processed_data, clean_raw_data)
from configs.logger_config import logger, error_logger
from configs.config import (RAW_S3_KEY, RAW_DATA_PATH, S3_RAW_BUCKET,
                            LAST_ROW_INDEX, BATCH_SIZE,
                             SOURCE_BUCKET, SOURCE_S3_KEY,
                             AWS_ACCESS_KEY, AWS_SECRET_KEY, 
                            TRANSFORMED_BUCKET, 
                            LAST_ROW_INDEX, 
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
from datetime import datetime
from airflow.models import Variable


def extract_and_validate_source_data(
        aws_access_key: str,
        aws_secret_access: str,
        source_bucket: str,
        source_s3_key: str,
        batch_size: int,
        last_row_index: int,
        raw_data_schema: Dict[str, str]
) -> Dict:
    """
    This function extracts the raw data from the given path and validates it

    Args:
        raw_data_path (str): Path to the raw data file
    Returns:
        Dict: A dictionary containing the status, message, and raw data as a pandas DataFrame
    """
    try:
        if not all(isinstance(param, str) for param in [aws_access_key,
                                                         aws_secret_access, 

                                                         source_bucket,
                                                         source_s3_key
                                                         ]):
            raise ValueError("All AWS parameters must be of type string")

        if not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if not isinstance(raw_data_schema, dict):
            raise ValueError("raw_data_schema must be a dictionary")
        if not isinstance(last_row_index, int):
            raise ValueError("last_row_index must be an integer")

        extract_source_data_status = extract_source_data(
            aws_access_key,
            aws_secret_access,
                source_bucket,
                source_s3_key,
                batch_size,
                last_row_index,
                raw_data_schema
            )
        if extract_source_data_status['status'] != 'success':
            raise Exception({
                "status": "error",
                "message": "An error occurred while extracting source data",
                "error": extract_source_data_status['error']
            })
        raw_df = extract_source_data_status['data']
        validate_status = validate_raw_data(raw_df)

        if validate_status['status'] != 'success':
            raise Exception({
                "status": "error",
                "message": "Raw data validation failed",
                "error": validate_status['error']
            })

        logger.info({
            "status": "success",
            "message": "Successfully extracted and validated raw data"
        })

        return {
            "status": "success",
            "message": "Successfully extracted and validated raw data",
            "data": raw_df
        }

    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while extracting and validating raw data",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while extracting and validating raw data",
            "error": str(e)
        })


def load_raw_data_to_s3(
        aws_access_key: str,
        aws_secret_access: str,
        raw_bucket: str,
        raw_s3_key: str,
        source_bucket: str,
        source_s3_key: str,
        raw_data_schema: Dict[str, str]
) -> Dict: 
    """
    This function loads the raw data to the raw s3 bucket

    Args:
        aws_access_key (str): AWS access key
        aws_secret_access (str): AWS secret access key
        raw_bucket (str): Raw s3 bucket name
        raw_s3_key (str): Raw s3 key
        raw_data_path (str): path of the raw data 

    Returns:
        Dict: A dictionary containing the status and message of the operation
    """

    try:
        try:
            if not all(isinstance(param, str) for param in [aws_access_key,
                                                            aws_secret_access, 
                                                            raw_bucket, 
                                                            raw_s3_key,
                                                                source_bucket,
                                                                source_s3_key
                                                            ]):
                raise ValueError("All aws parameters must be of type string")

            if not isinstance(raw_data_schema, dict):
                raise ValueError("raw_data_schema must be a dictionary")
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_access
            )
            existing_raw_obj = s3_client.head_object(Bucket=raw_bucket, Key=raw_s3_key)

            if existing_raw_obj:
                logger.info({
                    "status": "success",
                    "message": f"Raw data already exists at s3://{raw_bucket}/{raw_s3_key}. Raw data load skipped."
                })
                return {
                    "status": "success",
                    "message": f"Raw data already exists at s3://{raw_bucket}/{raw_s3_key}. Raw data load skipped."
                }
        except ClientError as e:
            
            if 'NoSuchKey' in str(e):
                logger.info({
                    "status": "info",
                    "message": f"No existing raw data found at s3://{raw_bucket}/{raw_s3_key}. A new object will be created."
                })
            

            source_response = s3_client.get_object(Bucket=source_bucket, Key=source_s3_key)
            
            if not source_response or 'Body' not in source_response:
                raise Exception({
                    "status": "error",
                    "message": "No data found in the source s3 path",
                    "response": source_response
                })

            source_data = source_response['Body']

            raw_df = pd.read_csv(source_data)

            s3_client.put_object(
                    Bucket=raw_bucket,
                    Key=raw_s3_key,
                    Body=raw_df.to_csv(index=False)
                )

        logger.info({
            "status": "success",
            "message": f"Successfully loaded raw data to raw s3 bucket: {raw_bucket}, s3 key: {raw_s3_key}"
        })
        return {
            "status": "success",
            "message": f"Successfully loaded raw data to raw s3 bucket: {raw_bucket}, s3 key: {raw_s3_key}"
        }

    except Exception as e:
        error_logger.error({
            "status": "error",
            "message": "An error occurred while loading raw data to s3",
            "error": str(e)
        })
        raise Exception({
            "status": "error",
            "message": "An error occurred while loading raw data to s3",
            "error": str(e)
        })


class LoadToTransformedS3:
    def __init__(self, s3_config: Dict):
        """
        This class contains operations that load the cleaned bikeshare data to the transformed S3 bucket.
        It ensures deduplication and partitions by user_type and week.
        
        Partitioning Strategy:
        s3://bucket/cleaned_bikeshare/
        ├── user_type=casual/
        │   ├── year=2022/
        │   │   ├── week=50/
        │   │   │   └── batch_001.parquet
        └── user_type=member/
            ├── year=2022/
            │   ├── week=50/
            │   │   └── batch_002.parquet
        """
        self.s3_config = s3_config
        self.s3_client = boto3.client('s3',
                                      aws_access_key_id = s3_config['access_key'],
                                      aws_secret_access_key = s3_config['secret_key'])
        
        self.base_prefix = s3_config.get('base_prefix')
        self.transformed_bucket = s3_config['transformed_bucket']
        self.raw_bucket = s3_config['raw_bucket']
        self.raw_s3_key = s3_config['raw_s3_key']
        self.duplicate_tracker_prefix = f"{self.base_prefix}/metadata/processed_bikeshare"


    def generate_partition_path(self,
                                user_type: str,
                                 year: str,
                                 week: str) -> str:
        """
        Generate S3 partition path based on ride datetime
        
        Args:
            ride_datetime: ride creation datetime
        
        Returns:
            S3 partition path
        """
        if not isinstance(year, str) or not isinstance(week, str):
            raise ValueError("year must be a string and week must be a string")
        
        if not isinstance(user_type, str):
            user_type = user_type.lower()
            raise ValueError("user_type must be a string")
        
        return (
            "cleaned_bikeshare/"
            f"user_type={user_type}/"
            f"year={year}/"
            f"week={week}/"
        )
    
    def get_batch_filename(self, batch_id: str) -> str:
        """Generate batch filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{batch_id}_{timestamp}.parquet"
        

    def get_processed_ride_ids_from_metadata(self, partition_year: str) -> Set[str]:
        """
        Get processed ride IDs from metadata storage in the bucket (more efficient approach)
        
        Args:
            partition_year: Year of the partition to retrieve IDs for
            
        Returns:
            Set of processed ride IDs
        """
        try:
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
                "message": f"Retrieved {len(processed_ids)} processed ride IDs for partition year: {partition_year}",
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
                "message": f"Error retrieving processed ride IDs for partition year: {partition_year}",
                "error": str(e)
            })
            raise Exception({
                "status": "error",
                "message": f"Error retrieving processed ride IDs for partition year: {partition_year}",
                "error": str(e)
            })

    def update_processed_ride_ids_metadata(self, 
                                            partition_year: str, 
                                            new_ride_ids: Set[str]) -> Dict:
        """
        Update metadata with newly processed ride IDs
        
        Args:
            partition_year: Year of the partition to update
            new_ride_ids: Set of new ride IDs to add
        """
        try:
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
                "message": f"Updated processed ride IDs for {partition_year}",
                "updated_ids_count": len(all_ids)
            }
        
        except Exception as e:
            logger.error({
                "status": "error",
                "message": f"Error updating processed ride IDs for {partition_year}",
                "error": str(e)
            })
            return {
                "status": "error",
                "message": f"Error updating processed ride IDs for {partition_year}",
                "error": str(e)
            }

    
    def filter_duplicate_rides(self, rides_df: pd.DataFrame) -> Dict:
        """
        Filter out duplicate rides based on existing data in S3 partitions
        Args:
            rides_df: DataFrame with rides data
        Returns:

        """
        try:

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
                    "status": "error",
                    "message": "Input DataFrame is empty, no rides to filter",
                    "input_shape": rides_df.shape

                })
                return {
                    "status": "error",
                    "message": "Input DataFrame is empty, no rides to filter",
                    "input_shape": rides_df.shape
                }
        
        except Exception as e:
            error_logger.error({
                "status": "error",
                "message": "An error occured while filtering duplicates data",
                "error": str(e)
            })
            raise Exception(
                {
                    "status": "error", 
                    "message": "An error occured while filtering duplicates data",
                    "error": str(e)
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

            if new_rides_df.empty:
                logger.info(
                    {
                        "status": "success",
                        "message": "No data to upload, dataframe is empty",
                        "uploaded_paths": []
                    }
                )
                return {
                    "status": "success",
                    "message": "No data to upload, dataframe is empty",
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
                            batch_size: int,
                            last_row_index: int,
                            raw_data_schema: Dict) -> Dict:
        """
        This function processes the raw data and returns a cleaned and processed data
        Args:
            data (pd.DataFrame): Raw data as a pandas DataFrame
            last_row_index (int): The last row index to process

        Returns:
            Dict: A dictionary containing the status, message, and cleaned data as a pandas DataFrame

        """

        try:
    
            
            extract_raw_data_status = extract_source_data(
                self.s3_config['access_key'],
                self.s3_config['secret_key'],
                self.s3_config['raw_bucket'],
                self.s3_config['raw_s3_key'],
                batch_size,
                last_row_index,
                    raw_data_schema
                )
            if extract_raw_data_status['status'] != 'success':
                raise Exception({
                    "status": "error",
                    "message": "An error occurred while extracting raw data",
                    "error": extract_raw_data_status['error']
                })
            data = extract_raw_data_status['data']

            clean_data_status = clean_raw_data(data)
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
            #save this processed data to a parquet format in S3
            s3_key = f"processed_data/processed_bikeshare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
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


    def process_rides_with_partitioning(self, processed_s3_key: str) -> Dict:

        """
        Complete pipeline with partitioning and duplicate prevention
        
        Args:
            rides_df: Raw rides DataFrame
            redshift_table_name: Name of the Redshift table to load data into
        """
        try:
            if not isinstance(processed_s3_key, str):
                raise ValueError("processed s3 key must be a string type")

            processed_data_obj = self.s3_client.get_object(Bucket=self.s3_config['transformed_bucket'], Key=processed_s3_key)


            processed_df = pd.read_parquet(io.BytesIO(processed_data_obj['Body'].read()))

            start_time = datetime.now()

            print("starting to load staging data to s3, start_time: ", start_time)
            
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

            print("new_rides_df shape after filtering duplicates: ", new_rides_df.head())

            upload_data_to_s3 = self.upload_partitioned_data(new_rides_df)
            
            if upload_data_to_s3['status'] == 'error':
                logger.error({
                    "status": "error",
                    "message": f"Failed to upload partitioned data to S3: {upload_data_to_s3['message']}"
                })
                return upload_data_to_s3
        
            print("finished loading to staging bucket, time taken: ", datetime.now() -  start_time)
            
            for partition_year, group in new_rides_df.groupby('partition_year'):
                ride_ids = set(group['ride_id'].astype(str))
                self.update_processed_ride_ids_metadata(partition_year, ride_ids)

            logger.info({
                "status": "info",
                "message": "Successfully processed rides and uploaded with partitioning",
            })
            return {
                "status": "success",
                "message": "Successfully processed rides and uploaded with partitioning",
            }
    
        except Exception as e:
            error_logger.error({
                "status": "error",
                "message": "An error occured while trying to processing staging data to the staging bucket",
                "error": str(e)
            })

            raise Exception({
                "status": "error",
                "message": "An error occured while trying to processing staging data to the staging bucket",
                "error": str(e)
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
        """
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
        self.session = sessionmaker(autocommit=False, autoflush=False, bind = self.engine)()

    
    def load_from_stage_to_table(
            self,
    ) -> Dict:
        
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
                            $1:member_casual
                            FROM
                            @{self.database}.{BikeRide.__table_args__['schema']}.{self.stage_name}
                            )
                            PATTERN = '.*parquet'
                            ON_ERROR = 'ABORT_STATEMENT'
                            PURGE = FALSE
                            FORCE = FALSE
                            FILE_FORMAT = (
                            TYPE = 'parquet'
                        )
                        """
                        )
                            
                db.execute(sql_query)
                # db.commit()

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
            error_logger.error({
                "status": "error",
                "message": "An error occurred while loading data from stage to table",
                "error": str(e)
            })
            raise Exception({
                "status": "error",
                "message": "An error occurred while loading data from stage to table",
                "error": str(e)
            })


def extract_and_validate_source_task(**context) -> None:

    extract_and_validate_source_data(
        aws_access_key=AWS_ACCESS_KEY,
        aws_secret_access=AWS_SECRET_KEY,
        source_bucket=SOURCE_BUCKET,
        source_s3_key=SOURCE_S3_KEY,
        batch_size=BATCH_SIZE,
        last_row_index=LAST_ROW_INDEX,
        raw_data_schema=raw_data_schema

    )

def load_raw_data_to_s3_task(**context) -> None:
    load_raw_data_to_s3(
            AWS_ACCESS_KEY,
            AWS_SECRET_KEY,
            S3_RAW_BUCKET,
            RAW_S3_KEY, 
            SOURCE_BUCKET,
            SOURCE_S3_KEY,
            raw_data_schema
    )

def process_raw_data_task(**context) -> None:
    process_raw_data_obj = LoadToTransformedS3(s3_config = S3_CONFIG
    )   

    process_raw_data_status = process_raw_data_obj.process_raw_data(
        batch_size=BATCH_SIZE,
        last_row_index=LAST_ROW_INDEX,
        raw_data_schema=raw_data_schema
    )
    processed_s3_key = process_raw_data_status['processed_data_key']
    context['ti'].xcom_push(key='processed_s3_key', value=processed_s3_key)

def validate_processed_data_task(**context) -> None:
    ti = context['ti']
    processed_s3_key = ti.xcom_pull(key='processed_s3_key', task_ids='process_raw_data')
    processed_s3_key = str(processed_s3_key)
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
    processed_s3_key = str(processed_s3_key)
    load_processed_data_to_s3 = LoadToTransformedS3(S3_CONFIG).process_rides_with_partitioning(processed_s3_key)

    if load_processed_data_to_s3['status'] != 'success':
        raise Exception({
            "status": "error",
            "message": "Failed to upload processed data to transformed bucket",
            "error": load_processed_data_to_s3['error']
        })

def update_checkpoint_variable_task(**context) -> None:
    Variable.set("last_row_index", LAST_ROW_INDEX + BATCH_SIZE)

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