#!/bin/bash
# Runs once, on first container start (empty data volume).
# POSTGRES_DB (set to "airflow" via docker-compose) is Airflow's own metadata
# database. This script additionally creates the "ecommerce" warehouse
# database and applies the star-schema DDL from src/db/schema to it.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ecommerce;
    CREATE DATABASE mlflow;
EOSQL

for f in /docker-entrypoint-initdb.d/schema/*.sql; do
    echo "Applying $f to ecommerce"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname ecommerce -f "$f"
done
