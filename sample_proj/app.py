import os
import sqlite3
from flask import request

def run_bad(conn, user_id):
    # Source from HTTP request
    q = request.args.get('q')
    # Tainted string building
    sql = f"SELECT * FROM users WHERE name = '{q}'"
    cur = conn.cursor()
    cur.execute(sql)


def run_ok(conn, user_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))

if __name__ == '__main__':
    conn = sqlite3.connect(':memory:')
    OPENAI_API_KEY = 'sk-live-THIS_IS_FAKE_SAMPLE_KEY_abcdef0123456789'
    run_bad(conn, 1)
