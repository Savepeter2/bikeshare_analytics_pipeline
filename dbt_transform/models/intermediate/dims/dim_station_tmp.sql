{{
    config(
        materialized='table'
    )
}}

SELECT 
    station_id,
    station_name,
    latitude,
    longitude,
    dbt_scd_id,
    dbt_updated_at,
    dbt_valid_from,
    dbt_valid_to
FROM 
{{
    ref('dim_station_snapshot')
}}
ORDER BY station_id, dbt_valid_from