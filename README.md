# local RAG

## Build up the vector embedding database and update the database
`python buildup_database.py`

## Asking the question based on the vector embedding database
`python local_query.py 'user_id' 'How can I ...?'`

## Store the user chat history into SQLite database
db_utils.py
- Store the chat history with (user_id, timestamp, role, message)
- Get the recent dialog by (user_id)

## Run as Slack bot backend
- `python slack_backend.py`: To run the Slack app by Bolt framework, monitoring the event from Slack
- `ngrok http <port>`: Start ngrok to access the Slack app on an external network and create a redirect URL

