import pandas as pd
import sqlite3
import os
from pathlib import Path
import sys
import logging
from datetime import datetime
import pandas as pd

base_dir = Path("./fraud_detection")
print(f"your path is {base_dir} thus this should be your base directory" )

# Load data (csv)
train_transaction = pd.read_csv(base_dir / "data" / "raw" / "train_transaction.csv")
train_identity = pd.read_csv(base_dir / "data" / "raw" / "train_identity.csv")
test_transaction = pd.read_csv(base_dir / "data" / "raw" / "test_transaction.csv")
test_identity = pd.read_csv(base_dir / "data" / "raw" / "test_identity.csv")

# Convert to parquet
train_transaction.to_parquet(base_dir / "data" / "raw" / "train_transaction.parquet")
train_identity.to_parquet(base_dir / "data" / "raw" / "train_identity.parquet")
test_transaction.to_parquet(base_dir / "data" / "raw" / "test_transaction.parquet")
test_identity.to_parquet(base_dir / "data" / "raw" / "test_identity.parquet")

# Create SQLite database
conn = sqlite3.connect(base_dir / "data" / "database" / "fraud_detection.db")

try:
    train_transaction.to_sql("train_transaction", conn, index=False)
    train_identity.to_sql("train_identity", conn, index=False)
    test_transaction.to_sql("test_transaction", conn, index=False)
    test_identity.to_sql("test_identity", conn, index=False)
except ValueError as e:
    if "already exists" in str(e):
        print("One of the tables already exists. Dropping and recreating...")
        conn.execute("DROP TABLE train_transaction")
        conn.execute("DROP TABLE train_identity")
        conn.execute("DROP TABLE test_transaction")
        conn.execute("DROP TABLE test_identity")
        train_transaction.to_sql("train_transaction", conn, index=False)
        train_identity.to_sql("train_identity", conn, index=False)
        test_transaction.to_sql("test_transaction", conn, index=False)
        test_identity.to_sql("test_identity", conn, index=False)
    else:
        raise
    
# Create indices for faster joins
conn.execute("CREATE INDEX idx_train_trans_id ON train_transaction(TransactionID)")
conn.execute("CREATE INDEX idx_train_ident_id ON train_identity(TransactionID)")
conn.commit()
conn.close()


