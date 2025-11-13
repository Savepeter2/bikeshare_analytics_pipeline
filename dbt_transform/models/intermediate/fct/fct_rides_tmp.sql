{{ 
    config(
        materialized='incremental',
        unique_key='ride_id'
    )
}}

with src_rides as (

    select 
    ride_id, 
    rideable_type,
    cast(started_at as date) as started_date,
    cast(ended_at as date) as ended_date,
    start_station_id,
    start_station_name,
    end_station_id,
    end_station_name,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    member_casual

    from {{ ref('stg_rides') }}

),

fct_rides as (
    select src_r.ride_id,
        dim_ss.station_id as start_station_id,
        dim_es.station_id as end_station_id,
        dim_ds.date_id as date_id,
        dim_rt.rideable_type_id,
        dim_ut.user_type_id,
        src_r.start_lat as start_station_latitude,
        src_r.start_lng as start_station_longitude,
        src_r.end_lat as end_station_latitude,
        src_r.end_lng as end_station_longitude

    from src_rides src_r
    left join {{ ref('dim_rideable_type_tmp') }} dim_rt
    on src_r.rideable_type = dim_rt.rideable_type
    left join {{ ref('dim_user_type_tmp') }} dim_ut
    on src_r.member_casual = dim_ut.user_type
    left join {{ ref('dim_date_tmp') }} dim_ds
    on src_r.started_date = dim_ds.date
    left join {{ ref('dim_station_tmp') }} dim_ss
    on src_r.start_station_id = dim_ss.station_id
    left join {{ ref('dim_station_tmp') }} dim_es
    on src_r.end_station_id = dim_es.station_id

),
new_fct_rides as (

    select fr.*
    from fct_rides fr

    {% if is_incremental() %}

        left join {{ this }} existing_table
        on fr.ride_id = existing_table.ride_id
        where existing_table.ride_id is null

    {% endif %}

)

select * from new_fct_rides


