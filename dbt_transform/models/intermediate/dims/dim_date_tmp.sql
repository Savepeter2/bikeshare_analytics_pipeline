{{ 
    config(
        materialized='table'
    )
}}

WITH date_series AS (
    SELECT
        DATEADD(day, SEQ4(), '2020-01-01'::DATE) AS date
    FROM TABLE(GENERATOR(ROWCOUNT => 2555))
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['date']) }} AS date_id,
    date,
    YEAR(date) AS year,
    MONTH(date) AS month,
    MONTHNAME(date) AS month_name,
    WEEK(date) AS week,
    DAY(date) AS day,
    DAYNAME(date) AS day_name,
    DAYOFWEEKISO(date) AS day_of_week,
    CURRENT_TIMESTAMP()::TIMESTAMP_NTZ AS created_at,
    CURRENT_TIMESTAMP()::TIMESTAMP_NTZ AS updated_at
FROM date_series
ORDER BY date