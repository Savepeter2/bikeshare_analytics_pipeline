{% snapshot dim_station_snapshot %}

{{
    config(
      target_schema='snapshots',
      unique_key='station_id',
      strategy='check',
      check_cols=['station_name'],
      invalidate_hard_deletes=True
    )
}}

SELECT *
FROM {{ ref('stg_stations_current') }}

{% endsnapshot %}