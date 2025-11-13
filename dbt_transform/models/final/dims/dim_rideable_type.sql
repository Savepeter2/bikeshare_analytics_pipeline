-- depends_on: {{ ref('dim_rideable_type_tmp') }}
{{
    config(
        materialized='table',
        post_hook=[
        "
        {% if execute and adapter.get_relation(
                database=target.database,
                schema=target.schema,
                identifier='dim_rideable_type_tmp'
            ) is not none %}

            DROP TABLE IF EXISTS {{ this }};
            DROP TABLE IF EXISTS DIM_RIDEABLE_TYPE_FINAL;
            ALTER TABLE DIM_RIDEABLE_TYPE_TMP RENAME TO DIM_RIDEABLE_TYPE_FINAL;
        {% else %}
            DROP TABLE IF EXISTS {{ this }};
        {% endif %}
        "
        ]
    )
}}

select
    1 as placeholder_column