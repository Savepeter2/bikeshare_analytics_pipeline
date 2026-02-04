FROM apache/airflow:2.9.2
USER root
RUN apt-get update && apt-get install -y git && apt-get clean
USER airflow
WORKDIR /opt/airflow
RUN pip uninstall -y apache-airflow-providers-google
RUN pip install poetry
COPY --chown=airflow:root pyproject.toml poetry.lock ./
RUN poetry lock && poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi
COPY --chown=airflow:root dbt_transform /opt/airflow/dbt_transform
WORKDIR /opt/airflow/dbt_transform
RUN poetry run dbt deps --target dev
WORKDIR /opt/airflow
ENV PYTHONPATH="/opt/airflow/src:/opt/airflow"