{{ 
    config(
        materialized='incremental',
        unique_key='ride_id',
        on_schema_change='fail',
        cluster_by = ['date_id','start_station_id','end_station_id','rideable_type_id','membership_type_id', 'started_time','ended_time']
    )
}}

with src_rides as (
    
    select 
    ride_id, 
    rideable_type,
    cast(started_at as date) as started_date,
    cast(ended_at as date) as ended_date,
    cast(started_at as time) as started_time,
    cast(ended_at as time) as ended_time,
    member_casual,
    start_station_id,
    end_station_id

    from {{ ref('stg_rides') }}

),

fct_rides as (
    select src_r.ride_id,
        dim_ss.station_id as start_station_id,
        dim_es.station_id as end_station_id,
        dim_ds.date_id as date_id,
        dim_rt.rideable_type_id,
        dim_ut.membership_type_id,
        src_r.started_time,
        src_r.ended_time

    from src_rides src_r
    left join {{ ref('dim_rideable_type_tmp') }} dim_rt
    on src_r.rideable_type = dim_rt.rideable_type
    left join {{ ref('dim_membership_type_tmp') }} dim_ut
    on src_r.member_casual = dim_ut.membership_type
    left join {{ ref('dim_date_tmp') }} dim_ds
    on src_r.started_date = dim_ds.date
    left join {{ ref('dim_station_tmp') }} dim_ss
    on src_r.start_station_id = dim_ss.station_id
    left join {{ ref('dim_station_tmp') }} dim_es
    on src_r.end_station_id = dim_es.station_id

),
new_fct_rides as (

    select fr.*,
    {% if is_incremental() %}
        COALESCE(existing_table.created_at, CURRENT_TIMESTAMP()::timestamp_ntz)
    {% else %}
        CURRENT_TIMESTAMP()::timestamp_ntz
    {% endif %}
    as created_at,
    CURRENT_TIMESTAMP()::timestamp_ntz as updated_at
    
    from fct_rides fr

    {% if is_incremental() %}

        left join {{ this }} existing_table
        on fr.ride_id = existing_table.ride_id
        where existing_table.ride_id is null

    {% endif %}

)

select * from new_fct_rides


