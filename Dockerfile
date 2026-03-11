FROM apache/airflow:3.1.7 
USER root     
RUN apt-get update && apt-get install -y git && apt-get clean
USER airflow
WORKDIR /opt/airflow
RUN pip uninstall -y apache-airflow-providers-google
RUN pip install apache-airflow-providers-fab==2.4.4
RUN pip install poetry
COPY --chown=airflow:root pyproject.toml poetry.lock ./
RUN poetry lock && poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi
RUN pip install "apache-airflow-providers-celery @ git+https://github.com/apache/airflow.git@ff33e4f#subdirectory=providers/celery"
RUN pip install apache-airflow-providers-amazon==9.22.0
RUN pip install "dbt-core>=1.9.0"
COPY --chown=airflow:root dbt_transform /opt/airflow/dbt_transform
COPY --chown=airflow:root src /opt/airflow/src           
COPY --chown=airflow:root configs /opt/airflow/configs
COPY --chown=airflow:root templates /opt/airflow/templates
WORKDIR /opt/airflow/dbt_transform
RUN rm -rf target dbt_packages && \
    poetry run dbt deps --target dev
WORKDIR /opt/airflow
ENV PYTHONPATH="/opt/airflow/src:/opt/airflow"