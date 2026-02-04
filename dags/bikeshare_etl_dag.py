import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from airflow import DAG
import pendulum
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.email import send_email
from src.etl import (
    get_batch_week_task,
    extract_and_validate_source_task,
    load_data_to_raw_bucket_task,
    process_raw_data_task,
    validate_processed_data_task,
    load_processed_data_to_s3_task,
    load_processed_s3_data_to_snowflake_task
)
from jinja2 import Template
from configs.config import AIRFLOW_MAIL_USERS, AIRFLOW_MAIL_SUBJECT_TEMPLATE, AIRFLOW_MAIL_HTML_TEMPLATE
from configs.logger_config import error_logger

def email_alert(context):
    """
    Handles email notifications for Success, Failure, and Retry states of the DAG.

    Args:
        context (dict): Airflow context dictionary containing task instance and other info.
    Returns:
        None
    """
    try:
        ti = context['task_instance']
        with open(AIRFLOW_MAIL_SUBJECT_TEMPLATE, 'r') as f:
            subject_template_content = f.read()

        with open(AIRFLOW_MAIL_HTML_TEMPLATE, 'r') as f:
            html_template_content = f.read()
        
        rendered_subject = Template(subject_template_content).render(**context)
        rendered_html_content = Template(html_template_content).render(**context)

        send_email(
            to=AIRFLOW_MAIL_USERS,
            subject=rendered_subject, 
            html_content=rendered_html_content,
            mime_subtype='html',
            context=context
        )
    except Exception as e:
        error_logger.error(
            {
                "status": "error",
                "message": f"Failed to send email notification of {ti.task_id}",
                "error": str(e)
            }
        )
        raise Exception({
            "status": "error",
            "message": f"Failed to send email notification of {ti.task_id}",
            "error": str(e)
        })


dag = DAG(
    'bikeshare_etl_dag',
    default_args={
        'owner': 'peter_de',
        'depends_on_past': False,
        'start_date': pendulum.datetime(2022, 12, 12, tz="Africa/Lagos"), #run for second week in december, - loads for week 48(1st week in december)
        # 'email': [AIRFLOW_MAIL_USERS],
        # 'on_success_callback': email_alert,
        # 'on_failure_callback': email_alert,
        # 'on_retry_callback': email_alert,
        'retries': 1,
        'max_active_runs': 1,
        'retry_delay': timedelta(minutes=2)
    },
    description='Weekly ETL for Capital Bikeshare Trips for December 2022',
    schedule_interval='0 10 * * 1',  # At 10:00 AM every Monday
    catchup=False,
    tags=['bikeshare', 'rides', 'pipeline', 'snowflake', 'dbt']
)

t1 = PythonOperator(
    task_id='get_batch_week',
    python_callable=get_batch_week_task,
    dag=dag
)

t2 = PythonOperator(
    task_id='extract_and_validate_source_data',
    python_callable=extract_and_validate_source_task,
    dag=dag
)

t3 = PythonOperator(
    task_id='load_data_to_raw_bucket',
    python_callable=load_data_to_raw_bucket_task,
    dag=dag
)

t4 = PythonOperator(
    task_id='process_raw_data',
    python_callable=process_raw_data_task,
    dag=dag
)

t5 = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed_data_task,
    dag=dag
)

t6 = PythonOperator(
    task_id='load_processed_data_to_s3',
    python_callable=load_processed_data_to_s3_task,
    dag=dag
)

t7 = PythonOperator(
    task_id='load_processed_s3_data_to_snowflake',
    python_callable=load_processed_s3_data_to_snowflake_task,
    dag=dag
)

dbt_run_staging = BashOperator(
    task_id='dbt_run_staging',
    bash_command='cd /opt/airflow/dbt_transform && poetry run dbt run -s staging --target dev',
    dag=dag
)

dbt_run_snapshots = BashOperator(
    task_id='dbt_snapshot',
    bash_command='cd /opt/airflow/dbt_transform && poetry run dbt snapshot --target dev',
    dag=dag
)

dbt_run_intermediate = BashOperator(
    task_id='dbt_run_intermediate',
    bash_command='cd /opt/airflow/dbt_transform && poetry run dbt run -s intermediate --target dev',
    dag=dag
)

dbt_test_intermediate = BashOperator(
    task_id='dbt_test_intermediate',
    bash_command='cd /opt/airflow/dbt_transform && poetry run dbt test -s intermediate --target dev',
    dag=dag
)

dbt_run_final = BashOperator(
    task_id='dbt_run_final',
    bash_command='cd /opt/airflow/dbt_transform && poetry run dbt run -s final --target dev',
    dag=dag
)


t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> dbt_run_staging >> dbt_run_snapshots >> dbt_run_intermediate >> dbt_test_intermediate >> dbt_run_final