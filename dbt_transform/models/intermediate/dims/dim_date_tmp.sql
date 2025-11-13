{{
    config(
        materialized='table'
    )
}}

WITH date_series AS (
    SELECT
        DATEADD(
            day,
            SEQ4(),
            '2020-01-01'::DATE
        ) AS date
    FROM TABLE(GENERATOR(ROWCOUNT => 2190)) -- Generates 5 years of dates (2020 - 2025) (365 * 5)
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['date']) }} AS date_id,
    date,
    DAYNAME(date) AS day_of_week
FROM date_series
ORDER BY date