{{
    config(
        materialized='table'
    )
}}

WITH src_membership_type AS (
    SELECT DISTINCT
        {{ dbt_utils.generate_surrogate_key(['member_casual']) }} AS membership_type_id,
        member_casual AS membership_type
    FROM {{ ref('stg_rides') }}
)

SELECT
    membership_type_id,
    membership_type,
    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at 
FROM src_membership_type
WHERE membership_type IS NOT NULL