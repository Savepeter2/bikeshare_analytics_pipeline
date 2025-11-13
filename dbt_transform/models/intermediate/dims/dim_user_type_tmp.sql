{{
    config(
        materialized='table'
    )
}}

WITH src_user_type AS (
    SELECT DISTINCT
        {{ dbt_utils.generate_surrogate_key(['member_casual']) }} AS user_type_id,
        member_casual AS user_type
    FROM {{ ref('stg_rides') }}
)

SELECT
    user_type_id,
    user_type 
FROM src_user_type
WHERE user_type IS NOT NULL