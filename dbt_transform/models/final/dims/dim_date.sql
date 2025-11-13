-- depends_on: {{ ref('dim_date_tmp') }}
{{
    config(
        materialized='table',
        post_hook=[
        "
        {% if execute and adapter.get_relation(
                database=target.database,
                schema=target.schema,
                identifier='dim_date_tmp'
            ) is not none %}

            DROP TABLE IF EXISTS {{ this }};
            DROP TABLE IF EXISTS DIM_DATE_FINAL;
            ALTER TABLE DIM_DATE_TMP RENAME TO DIM_DATE_FINAL;
        {% else %}
            DROP TABLE IF EXISTS {{ this }};
        {% endif %}
        "
        ]
    )
}}

select
    1 as placeholder_column