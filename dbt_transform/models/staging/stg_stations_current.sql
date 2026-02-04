-- This creates a view of the current stations from the rides data
-- This will be used as the source for the station snapshot, the stg_rides wasn't used because 
-- it is an incremental model and  all the data is needed for the snapshot
{{
    config(
        materialized='view'
    )
}}

WITH src_stations AS (
    SELECT 
        start_station_id AS station_id, 
        start_station_name AS station_name, 
        start_lat AS latitude, 
        start_lng AS longitude
    FROM BIKESHARE_DB.RAW.RAW_BIKE_RIDES
    UNION ALL
    SELECT 
        end_station_id AS station_id, 
        end_station_name AS station_name, 
        end_lat AS latitude, 
        end_lng AS longitude
    FROM BIKESHARE_DB.RAW.RAW_BIKE_RIDES
    WHERE end_station_id IS NOT NULL
)

SELECT DISTINCT 
    station_id,
    station_name,
    latitude,
    longitude
FROM src_stations
WHERE station_id IS NOT NULL