import os
import json
import subprocess
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Slack Bot Token (Make sure set up the environment variable)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_API_URL = "https://slack.com/api/chat.postMessage"


def run_python_script(query):
    """
    Run local_query.py and return
    """
    process = subprocess.Popen(
        ["python", "local_query.py", query],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = process.communicate()

    if error:
        return f"Runtime error: {error.decode('utf-8')}"

    return output.decode("utf-8")


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Receive Slack message, Run local_query.py, and return answer
    """
    data = request.json
    print(data)

    # Slack initial verify
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # If the event type is message
    if "event" in data and data["event"]["type"] == "message":
        user_text = data["event"]["text"]
        channel_id = data["event"]["channel"]
        user_id = data["event"]["user"]

        # Avoid bot reply itself
        if "bot_id" in data["event"]:
            return jsonify({"status": "ignored"})

        # Run Python script to response
        response_text = run_python_script(user_text)

        # 回Return result to Slack
        requests.post(
            SLACK_API_URL,
            json={"channel": channel_id, "text": response_text},
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        )

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
