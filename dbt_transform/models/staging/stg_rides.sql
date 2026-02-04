{{
  config(
        materialized='incremental',
        unique_key='ride_id',
        on_schema_change='fail',
        cluster_by = ['ride_id','rideable_type','member_casual']
    )
}}

SELECT raw_r.ride_id, 
        raw_r.rideable_type,
        raw_r.started_at,
        raw_r.ended_at,
        raw_r.member_casual,
        raw_r.start_station_id,
        raw_r.end_station_id,
        {% if is_incremental() %}
            COALESCE(existing_table.created_at, CURRENT_TIMESTAMP()::timestamp_ntz)
        {% else %}
            CURRENT_TIMESTAMP()::timestamp_ntz
        {% endif %}
         as created_at,
        CURRENT_TIMESTAMP()::timestamp_ntz as updated_at

FROM BIKESHARE_DB.RAW.RAW_BIKE_RIDES raw_r

{% if is_incremental() %}

    LEFT JOIN {{ this }} existing_table
    ON raw_r.ride_id = existing_table.ride_id
    WHERE existing_table.ride_id IS NULL

{% endif %}