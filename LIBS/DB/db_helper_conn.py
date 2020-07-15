import os
import MySQLdb
import json

def get_db_conn():
    json_file = os.path.join(os.path.abspath('.'), 'db_config.json')
    data = json.load(json_file)

    db = MySQLdb.connect(data['host'], data['username'], data['password'],
                         data['database'], use_unicode=True, charset='utf8')

    return db


