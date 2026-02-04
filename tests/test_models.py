import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch, MagicMock
from src.models import BikeRide
from src.models import create_session
from sqlalchemy import String, Float
from snowflake.sqlalchemy import TIMESTAMP_NTZ

@pytest.fixture(scope="class")
def sqlalchemy_config() -> dict:
    """
    Fixture to provide SQLAlchemy configuration for tests.
    """
    return {
        "connection_string": "snowflake://test_user:test_password@account/db/schema?warehouse=wh&role=role"
    }

class TestCreateSession:
    """
    Tests the create_session context manager.
    """
    @patch('src.models.Base')
    @patch('src.models.sessionmaker')
    @patch('src.models.create_engine')
    def test_create_session_success(self,
                                    mock_create_engine: MagicMock,
                                    mock_session_maker: MagicMock,
                                    mock_base_class: MagicMock,
                                    sqlalchemy_config: dict) -> None:
        
        """
        Tests successful creation of a session.
        This ensures that the create_engine instance is created and session is yielded properly.
        
        Args:
            mock_create_engine (MagicMock): Mock for create_engine function.
            mock_session_maker (MagicMock): Mock for sessionmaker function.
            mock_base_class (MagicMock): Mock for Base class.
            sqlalchemy_config (dict): SQLAlchemy configuration fixture.            

        Returns:
            None
        """
        valid_conn_string = sqlalchemy_config['connection_string']
        
        mock_engine_instance = MagicMock()
        mock_create_engine.return_value = mock_engine_instance

        mock_session_instance = MagicMock()
        mock_session_maker.return_value.return_value = mock_session_instance

        create_session_instance = create_session(valid_conn_string)

        with create_session_instance as session:
            assert session == mock_session_instance
            mock_create_engine.assert_called_once_with(valid_conn_string)
            mock_session_maker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine_instance)
            mock_base_class.metadata.create_all.assert_called_once_with(mock_engine_instance)
    
    @patch('src.models.create_engine')
    def test_create_engine_failure(self,
                                    mock_create_engine: MagicMock) -> None:
        """
        Tests failure during engine creation.
        This ensures that the appropriate exception is raised when engine creation fails.

        Args:
            mock_create_engine (MagicMock): Mock for create_engine function.
            sqlalchemy_config (dict): SQLAlchemy configuration fixture.
        Returns:
            None
        """
        mock_create_engine.side_effect = Exception("Engine creation failed")
        invalid_conn_string = "invalid_connection_string"
        
        with pytest.raises(Exception) as excinfo:
            create_session_instance = create_session(invalid_conn_string)
            with create_session_instance as session:
                pass

        assert str(excinfo.value) == "Engine creation failed"
        mock_create_engine.assert_called_once_with(invalid_conn_string)

    @patch('src.models.sessionmaker')
    @patch('src.models.create_engine')
    def test_session_rollback_on_exception(self,
                                            mock_create_engine: MagicMock,
                                            mock_session_maker: MagicMock,
                                            sqlalchemy_config: dict
                                           ) -> None:
        """
        Tests that the session is rolled back on any exception that occurs when creating a session object or during its usage.
        It also ensures that the appropriate exception is raised. It ensures that the session is closed in the finally block.
        
        Args:
            mock_create_engine (MagicMock): Mock for create_engine function.
            mock_session_maker (MagicMock): Mock for sessionmaker function.
            sqlalchemy_config (dict): SQLAlchemy configuration fixture.
        
        Returns:
            None
        """
        mock_engine_instance = MagicMock()
        mock_create_engine.return_value = mock_engine_instance
        mock_session_instance = MagicMock()
        mock_session_maker.return_value.return_value = mock_session_instance
        valid_conn_string = sqlalchemy_config['connection_string']
        
        with pytest.raises(Exception) as excinfo:
            create_session_instance = create_session(valid_conn_string)
            with create_session_instance as session:
                raise Exception("Exception during session usage")
        assert str(excinfo.value) == "Exception during session usage"

        mock_create_engine.assert_called_once_with(valid_conn_string)
        mock_session_maker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine_instance)
        mock_session_instance.rollback.assert_called_once()
        mock_session_instance.close.assert_called_once()

    @patch('src.models.Base')
    @patch('src.models.sessionmaker')
    @patch('src.models.create_engine')
    def test_session_close_always_called(self,
                                mock_create_engine: MagicMock,
                                mock_session_maker: MagicMock,
                                mock_base_class: MagicMock, 
                                 sqlalchemy_config: dict) -> None:
        """
        Tests that the session is always closed in the finally block, even if no exceptions occur.
        
        Args:
            sqlalchemy_config (dict): SQLAlchemy configuration fixture.
        Returns:
            None
        """
        mock_engine_instance = MagicMock()
        mock_create_engine.return_value = mock_engine_instance
        mock_session_instance = MagicMock()
        mock_session_maker.return_value.return_value = mock_session_instance
        mock_base_class.metadata.create_all.return_value = None
        valid_conn_string = sqlalchemy_config['connection_string']
        with create_session(valid_conn_string) as session:
            pass
        mock_base_class.metadata.create_all.assert_called_once_with(mock_engine_instance)
        mock_session_instance.close.assert_called_once()

class TestBikeRideModel:
    """
    Tests for the BikeRide ORM model.
    It mainly focuses on ensuring that the model's attributes are correctly defined.
    Protects against accidental changes to the model structure.

    Args:
        None

    Returns:
        None
    """

    def test_bike_ride_model_metadata(self) -> None:
        """
        Tests that the BikeRide model has the correct table name and schema defined.

        Args:
            None
        
        Returns:
            None
        """
        assert BikeRide.__tablename__ == 'raw_bike_rides'
        assert BikeRide.__table_args__['schema'] == 'RAW'



    def test_bike_ride_model_attributes(self) -> None:
        """
        Tests that the BikeRide model has the correct attributes defined.
        
        Returns:
            None
        """
        
        expected_attributes = {
            'ride_id',
            'rideable_type',
            'started_at',
            'ended_at',
            'start_station_name',
            'start_station_id',
            'end_station_name',
            'end_station_id',
            'start_lat',
            'start_lng',
            'end_lat',
            'end_lng',
            'member_casual'
        }
        bike_ride_attributes = {attr.key for attr in BikeRide.__table__.columns}
        assert bike_ride_attributes == expected_attributes
    
    def test_bike_ride_model_primary_key(self) -> None:
        """
        Tests that the primary key of the BikeRide model is correctly defined.
        This ensures that the 'ride_id' attribute is set as the primary key.

        Args:
            None
        
        Returns:
            None
        """
        primary_keys = [key.name for key in BikeRide.__table__.primary_key.columns]
        assert primary_keys == ['ride_id']

    def test_bike_ride_model_repr(self) -> None:
        """
        Tests the __repr__ method of the BikeRide model.
        It ensures that the string representation is as expected.

        Args:
            None
        
        Returns:
            None
        """
        bike_ride = BikeRide(
            ride_id='test_ride_id',
            started_at='2024-01-01 10:00:00'
        )
        expected_repr = "<BikeRide(ride_id='test_ride_id', started_at='2024-01-01 10:00:00')>"
        assert repr(bike_ride) == expected_repr
    
    def test_bike_ride_model_column_types(self) -> None:
        """
        Tests that the BikeRide model has the correct column types defined.
        
        Returns:
            None
        """

        column_types = {
            'ride_id': String,
            'rideable_type': String,
            'started_at': TIMESTAMP_NTZ,
            'ended_at': TIMESTAMP_NTZ,
            'start_station_name': String,
            'start_station_id': String,
            'end_station_name': String,
            'end_station_id': String,
            'start_lat': Float,
            'start_lng': Float,
            'end_lat': Float,
            'end_lng': Float,
            'member_casual': String
        }

        for column_name, expected_type in column_types.items():
            actual_type = type(BikeRide.__table__.columns[column_name].type)
            assert actual_type == expected_type, f"Column '{column_name}' expected type {expected_type}, got {actual_type}"
