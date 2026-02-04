from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from snowflake.sqlalchemy import TIMESTAMP_NTZ
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
import pandas as pd
import logging


Base = declarative_base()

@contextmanager
def create_session(
connection_string: str,
session: sessionmaker = None
):
    try:
        engine = create_engine(connection_string)
        Session = sessionmaker( autocommit=False, autoflush=False, bind = engine)
        Base.metadata.create_all(engine)
        session = Session()
        yield session
    
    except Exception as e:
        if session:
            session.rollback()
        raise e 
    
    finally:
        if session:
            session.close()


class BikeRide(Base):
    """
    ORM model for the bike rides staging table in Snowflake.
    """
    __tablename__ = 'raw_bike_rides'
    __table_args__ = {
        'schema': 'RAW', #will still change, has to be created at and updated at
        'comment': 'Bike sharing ride data from transformed S3 bucket'
    }

    ride_id = Column(String(255), primary_key=True)
    rideable_type = Column(String(255))
    started_at = Column(TIMESTAMP_NTZ)
    ended_at = Column(TIMESTAMP_NTZ)
    start_station_name = Column(String(255))
    start_station_id = Column(String(255))
    end_station_name = Column(String(255))
    end_station_id = Column(String(255))
    start_lat = Column(Float)
    start_lng = Column(Float)
    end_lat = Column(Float)
    end_lng = Column(Float)
    member_casual = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<BikeRide(ride_id='{self.ride_id}', started_at='{self.started_at}')>"


class AlertsLog(Base):
    """
    ORM model for persisting real-time flagged alerts in Snowflake.
    """
    __tablename__ = 'alerts_log'
    __table_args__ = {
        'schema': 'RAW',
        'comment': 'Logs for alerts generated during data processing'
    }

    id = Column(String(64), primary_key=True)
    ride_id = Column(String(255), unique = True, nullable=False)
    flag_type = Column(String(255), nullable=False)
    rideable_type = Column(String(255), nullable=False)
    started_at = Column(TIMESTAMP_NTZ, nullable=False)
    ended_at = Column(TIMESTAMP_NTZ, nullable=False)
    start_station_name = Column(String(255), nullable=False)
    end_station_name = Column(String(255))
    start_station_id = Column(Integer)
    end_station_id = Column(Integer)
    start_lat = Column(Float, nullable=False)
    start_lng = Column(Float, nullable=False)
    end_lat = Column(Float)
    end_lng = Column(Float)
    member_casual = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP_NTZ, default=datetime.now)
    updated_at = Column(TIMESTAMP_NTZ, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<AlertsLog(id='{self.id}', ride_id='{self.ride_id}', flag_type='{self.flag_type}', started_at='{self.started_at}', created_at='{self.created_at}')>"

