import sys
import pymysql
import logging
db = pymysql.connect(host='localhost', port=3306, user='root', password='**', database='xh')


class msql:
    def init(self):
        self.cursor = db.cursor()
        self.conn = db

    def create_mysql_table(self, table_name):
        # Create mysql table if not exists
        sql = "create table if not exists " + table_name + "(milvus_id TEXT, image_id TEXT, image_data MEDIUMBLOB not null );"
        try:
            self.cursor.execute(sql)
            logging.debug(f"MYSQL create table: {table_name} with sql: {sql}")
        except Exception as e:
            logging.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)

    def load_data_to_mysql(self, table_name, data):
        # Batch insert (Milvus_ids, img_path) to mysql
        sql = "insert into " + table_name + " (milvus_id,image_id, image_data) values (%s,%s, %s);"
        try:
            self.cursor.executemany(sql, data)
            self.conn.commit()
            logging.debug(f"MYSQL loads data to table: {table_name} successfully")
        except Exception as e:
            logging.error(f"MYSQL ERROR: {e} with sql: {sql}")
            sys.exit(1)
