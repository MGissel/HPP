import sqlite3

def init_db(path:str, table_name: str, table: dict) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(_create_table_text(table_name, table))
    con.commit()
    return con

def _create_table_text(table_name: str, columns: dict) -> str:
    cols = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
    print(cols)
    return f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})"
