# local RAG

## Applications that need to Installed first
- Git
- VScode
- Conda
- Ollama
- ngrok


On Windows will need install MSVC C++14.0 development toolkits

## Create New Environment
`conda create --name <env_name> python=3.12`

`conda activate <env_name>`

## Install Python dependency
`while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt`

## Build up the vector embedding database and update the database
`python buildup_database.py`

## Asking the question based on the vector embedding database (for testing)
`python local_query.py 'user_id' 'How can I ...?'`

## Store the user chat history into SQLite database
db_utils.py
- Store the chat history with (user_id, timestamp, role, message)
- Get the recent dialog by (user_id)

## Run Ollama
`ollama pull mxbai-embed-large`

`ollama pull llama3.1`

`ollama serve`

## Run as Slack bot backend
- `python slack_backend.py`: To run the Slack app by Bolt framework, monitoring the event from Slack
- `ngrok http <port>`: Start ngrok to access the Slack app on an external network and create a redirect URL

## If you change the ngrok url, you need to update the Event Subscriptions url in slack App page
Change the request URL to:
`https://<your_ngrok_url>.ngrok-free.app/slack/events`
