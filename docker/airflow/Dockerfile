FROM apache/airflow:2.9.3-python3.12
# Project + ML dependencies baked into the image (pinned to match requirements.txt where the
# Airflow Python version allows) so containers start fast and identically every time.
RUN pip install --no-cache-dir pandas==2.1.4 openpyxl==3.1.5 requests==2.32.3 \
    scikit-learn==1.9.1 xgboost==3.4.1 mlflow-skinny==3.16.1 matplotlib==3.11.2 skops==0.16.0 \
 && pip check
