{{
    config(
        materialized='table'
    )
}}

WITH src_rideable_type AS (
    SELECT DISTINCT {{ dbt_utils.generate_surrogate_key(['rideable_type']) }} AS rideable_type_id,
        rideable_type
    FROM {{ ref('stg_rides') }}
)

SELECT
    rideable_type_id,
    rideable_type
FROM src_rideable_type
