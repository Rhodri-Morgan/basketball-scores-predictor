import pandas as pd
import sqlite3
import os
import shutil


DATABASE_PATH = os.path.join(os.getcwd(), 'basketball.sqlite')
OUT_PATH = os.path.join(os.getcwd(), 'extracted_raw_data')


if __name__ == '__main__':
    if not os.path.exists(DATABASE_PATH):
        print('Please add basketball.sqlite')
        exit(1)

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    # Get tables
    cursorObj = connection.cursor()
    cursorObj.execute('SELECT name from sqlite_master where type= "table"')
    list_of_tables = cursorObj.fetchall()
    list_of_tables = [table_name[0].lower() for table_name in list_of_tables]

    # Save data
    for i, table_name in enumerate(list_of_tables):
        i += 1
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', connection)
        df.to_parquet(os.path.join(OUT_PATH, f'{table_name}.parquet'))
        print(f'{i} / {len(list_of_tables)} - Outputted Table {table_name}')
