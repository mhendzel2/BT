import os
import glob
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import streamlit as st

# Default local connection string - user can override
DEFAULT_DB_URL = "postgresql://postgres:password@localhost:5432/market_data"

def get_engine(db_url=DEFAULT_DB_URL):
    return create_engine(db_url)

def init_db(engine):
    with engine.connect() as conn:
        # Create table for tracking imported files
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS imported_files (
                filename TEXT PRIMARY KEY,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lowercase and remove underscores and spaces to standardize column names
    df.columns = [c.lower().replace("_", "").replace(" ", "") for c in df.columns]
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    return df

def ingest_folder(folder_path: str, engine):
    if not os.path.exists(folder_path):
        return 0, []

    files = glob.glob(os.path.join(folder_path, "*.csv"))
    imported_count = 0
    errors = []

    # Get list of already imported files
    with engine.connect() as conn:
        result = conn.execute(text("SELECT filename FROM imported_files"))
        imported_files = {row[0] for row in result}

    for f in files:
        filename = os.path.basename(f)
        if filename in imported_files:
            continue

        try:
            # Try reading with default settings first
            try:
                df = pd.read_csv(f)
                if len(df.columns) <= 1:
                     df = pd.read_csv(f, sep=None, engine='python')
            except:
                df = pd.read_csv(f, sep=None, engine='python')

            df = _normalize_cols(df)
            
            # Identify type
            cols = set(df.columns)
            table_name = None
            
            if "callpremium" in cols or "putpremium" in cols:
                table_name = "stock_screener"
            elif ("oichange" in cols or "oi_change" in cols) and "strike" in cols:
                table_name = "chain_oi"
            
            if table_name:
                # Ensure date column is datetime
                for date_col in ["date", "trade_date", "asofdate", "timestamp"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                        # Rename to standard 'date'
                        if date_col != "date":
                            df = df.rename(columns={date_col: "date"})
                        break
                
                # Add filename for lineage
                df["source_file"] = filename
                
                # Write to DB
                df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)
                
                # Mark as imported
                with engine.connect() as conn:
                    conn.execute(text("INSERT INTO imported_files (filename) VALUES (:fn)"), {"fn": filename})
                    conn.commit()
                
                imported_count += 1
            
        except Exception as e:
            errors.append(f"Error importing {filename}: {str(e)}")

    return imported_count, errors

def load_data_from_db(engine, start_date=None, end_date=None):
    query_params = {}
    date_filter = ""
    
    if start_date:
        date_filter += " AND date >= :start_date"
        query_params["start_date"] = start_date
    if end_date:
        date_filter += " AND date <= :end_date"
        query_params["end_date"] = end_date

    try:
        stock_df = pd.read_sql(text(f"SELECT * FROM stock_screener WHERE 1=1 {date_filter}"), engine, params=query_params)
    except Exception:
        stock_df = pd.DataFrame()

    try:
        chain_df = pd.read_sql(text(f"SELECT * FROM chain_oi WHERE 1=1 {date_filter}"), engine, params=query_params)
    except Exception:
        chain_df = pd.DataFrame()
        
    return stock_df, chain_df
