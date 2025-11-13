FROM apache/airflow:2.9.2
USER root
RUN apt-get update && apt-get install -y git && apt-get clean
USER airflow
WORKDIR /opt/airflow
RUN pip uninstall -y apache-airflow-providers-google
RUN pip install --no-cache-dir poetry
COPY --chown=airflow:root pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi
ENV PYTHONPATH="/opt/airflow/src:/opt/airflow"
