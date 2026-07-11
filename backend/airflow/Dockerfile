FROM apache/airflow:3.3.0

COPY requirements.txt /requirements.txt
# The base image supplies AIRFLOW_VERSION. Pinning it here prevents pip from
# silently selecting a different Airflow release while installing providers.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
