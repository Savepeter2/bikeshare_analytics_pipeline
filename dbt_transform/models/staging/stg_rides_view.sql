{{
  config(
        materialized='view',
    )
}}

SELECT raw_r.ride_id, 
        raw_r.rideable_type,
        raw_r.started_at,
        raw_r.ended_at,
        raw_r.start_station_id,
        raw_r.start_station_name,
        raw_r.end_station_id,
        raw_r.end_station_name,
        raw_r.start_lat,
        raw_r.start_lng,
        raw_r.end_lat,
        raw_r.end_lng,
        raw_r.member_casual,
        current_timestamp() AS last_updated_at

FROM BIKESHARE_DB.RAW.RAW_BIKE_RIDES raw_r