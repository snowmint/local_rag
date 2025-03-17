import sqlite3
from datetime import datetime

DB_FILE = "chat_history.db"


def init_db():
    """
    Init SQLite database, create data table
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def store_message(user_id, message, role="user"):
    """
    Store user and bot dialog
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()
    cursor.execute("INSERT INTO messages (user_id, timestamp, role, message) VALUES (?, ?, ?, ?)",
                   (user_id, timestamp, role, message))

    conn.commit()
    conn.close()


def get_recent_messages(user_id, limit=10):
    """
    Get recent dialog of user
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp, role, message FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                   (user_id, limit))

    rows = cursor.fetchall()
    conn.close()

    return [{"timestamp": row[0], "role": row[1], "message": row[2]} for row in reversed(rows)]
