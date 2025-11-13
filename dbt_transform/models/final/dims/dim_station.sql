-- depends_on: {{ ref('dim_station_tmp') }}
{{
    config(
        materialized='table',
        post_hook=[
        "
        {% if execute and adapter.get_relation(
                database=target.database,
                schema=target.schema,
                identifier='dim_station_tmp'
            ) is not none %}

            DROP TABLE IF EXISTS {{ this }};
            DROP TABLE IF EXISTS DIM_STATION_FINAL;
            ALTER TABLE DIM_STATION_TMP RENAME TO DIM_STATION_FINAL;

        {% else %}
            DROP TABLE IF EXISTS {{ this }};
        {% endif %}
        "
        ]
    )
}}

select
    1 as placeholder_column