"""Generic Postgres upsert helper shared by load_warehouse and rfm_features."""
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy.engine import Engine


def _to_native(value):
    """Converts pandas/numpy scalars (Int64 NA, numpy.int64, NaT, ...) to
    plain Python types psycopg2 knows how to adapt."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _rows(df: pd.DataFrame) -> list:
    return [tuple(_to_native(v) for v in r) for r in df.itertuples(index=False, name=None)]


def read_sql_df(engine: Engine, sql: str) -> pd.DataFrame:
    """Runs a read-only query via a raw psycopg2 cursor rather than
    pandas.read_sql(engine, ...): pandas' SQLAlchemy-engine detection is
    version-sensitive and raises "'Engine' object has no attribute
    'cursor'" against the older SQLAlchemy Airflow 2.9 bundles."""
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def bulk_insert(engine: Engine, schema: str, table: str, df: pd.DataFrame):
    """Plain append-only insert. Bypasses pandas.to_sql() deliberately: it
    detects SQLAlchemy engines by version-sensitive introspection that
    breaks against older SQLAlchemy builds (e.g. the one Airflow 2.9
    bundles), raising "'Engine' object has no attribute 'cursor'"."""
    if df.empty:
        return
    cols = list(df.columns)
    sql = f"INSERT INTO {schema}.{table} ({', '.join(cols)}) VALUES %s"

    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        execute_values(cur, sql, _rows(df))
        conn.commit()
    finally:
        conn.close()


def upsert(engine: Engine, schema: str, table: str, df: pd.DataFrame, conflict_cols: list, update_cols: list | None = None):
    if df.empty:
        return
    cols = list(df.columns)

    conflict_clause = f"ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
    if update_cols:
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        conflict_clause = f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {set_clause}"

    sql = f"INSERT INTO {schema}.{table} ({', '.join(cols)}) VALUES %s {conflict_clause}"

    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        execute_values(cur, sql, _rows(df))
        conn.commit()
    finally:
        conn.close()
