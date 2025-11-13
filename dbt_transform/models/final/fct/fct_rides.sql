-- depends_on: {{ ref('fct_rides_tmp') }}
{{
    config(
        materialized='table',
        post_hook=[
        "
        {% if execute and adapter.get_relation(
                database=target.database,
                schema=target.schema,
                identifier='fct_rides_tmp'
            ) is not none %}
            DROP TABLE IF EXISTS {{ this }};
            DROP TABLE IF EXISTS FCT_RIDES_FINAL;
            ALTER TABLE FCT_RIDES_TMP RENAME TO FCT_RIDES_FINAL;
        {% else %}
            DROP TABLE IF EXISTS {{ this }};
        {% endif %}
        "
        ]
    )
}}

select
    1 as placeholder_column