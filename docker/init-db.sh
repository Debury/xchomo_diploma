#!/bin/bash
# Creates the climate_app database alongside the existing dagster database.
# Mounted as an init script in the dagster-postgres container.

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE climate_app'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'climate_app')\gexec
    GRANT ALL PRIVILEGES ON DATABASE climate_app TO $POSTGRES_USER;
EOSQL
