import redis
import json
from datetime import datetime

# 連接 Redis
redis_client = redis.StrictRedis(
    host="localhost", port=6379, db=0, decode_responses=True)


def store_message(user_id, message, role="user"):
    """
    Store bot and user's dialog to Redis
    """
    key = f"user:{user_id}:messages"
    timestamp = datetime.now().isoformat()
    new_entry = {"timestamp": timestamp, "role": role, "message": message}

    existing_logs = redis_client.get(key)
    logs = json.loads(existing_logs) if existing_logs else []

    logs.append(new_entry)
    logs = logs[-50:]  # 最多保留 50 條歷史訊息
    redis_client.set(key, json.dumps(logs))


def get_recent_messages(user_id, number=10):
    """
    Get recent user dialog
    """
    key = f"user:{user_id}:messages"
    existing_logs = redis_client.get(key)

    if existing_logs:
        logs = json.loads(existing_logs)
        return logs[-number:]  # get least number's dialog
    return []
