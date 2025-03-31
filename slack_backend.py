import logging  # For better debug
from slack_bolt import App, Say, BoltContext
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
# from redis_utils import store_message # Cache storage, may not fit for store long term
from db_utils import store_message, init_db  # use SQLite
from langdetect import detect
import langid
from googletrans import Translator  # not good enough
import asyncio

# from slack_sdk import WebClient
logging.basicConfig(level=logging.DEBUG)

# Remember to add environment variable first
# export SLACK_SIGNING_SECRET=***
# export SLACK_BOT_TOKEN=xoxb-***
# export SLACK_APP_TOKEN=xapp-***
IT_SUPPORT_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeF6dozhuoOPDxDhj6ldkN5RmTZjUa8maYATfHjUsFGtliPIA/viewform?pli=1"
ASSET_REGISTRATION_URL = "https://airtable.com/appaL6KyixMc59fZW/shrXBQYLiTR99nTkP"

IT_SUPPORT_FORM = f"<{IT_SUPPORT_URL}|IT Support>"
ASSET_REGISTRATION_FORM = f"<{ASSET_REGISTRATION_URL}|Asset Registration>"


init_db()

app = App()
executor = ThreadPoolExecutor(max_workers=5)

# To see CPU core count
# import multiprocessing
# cpu_amount = multiprocessing.cpu_count()
# print(cpu_amount)IT_SUPPORT_LINK

# import os
# cpu_amount = os.cpu_count()
# print(cpu_amount)

PROTECTED_TERMS = {
    "IT support": "IT_SUPPORT",
    "Asset registration": "ASSET_REGISTRATION"
}

PROTECTED_URLS = {
    IT_SUPPORT_URL: "IT_SUPPORT_LINK",
    ASSET_REGISTRATION_URL: "ASSET_REGISTRATION_LINK"
}


def protect_text(text):
    for term, placeholder in PROTECTED_TERMS.items():
        text = re.sub(re.escape(term), placeholder, text, flags=re.IGNORECASE)

    for url, placeholder in PROTECTED_URLS.items():
        # Use "replace" to keep URL format
        text = text.replace(url, placeholder)
    return text


def restore_text(text):
    support_form_replace = re.compile(
        re.escape("IT_SUPPORT_LINK"), re.IGNORECASE)
    text = support_form_replace.sub(f"<{IT_SUPPORT_URL}|IT Support>", text)

    asset_form_replace = re.compile(
        re.escape("ASSET_REGISTRATION_LINK"), re.IGNORECASE)
    text = asset_form_replace.sub(
        f"<{ASSET_REGISTRATION_URL}|Asset Registration>", text)

    for term, placeholder in PROTECTED_TERMS.items():
        text = re.sub(re.escape(placeholder), term, text, flags=re.IGNORECASE)

    return text


async def translate_to_english(text):
    """
    Detect the language and translate to english
    """
    detected_lang = langid.classify(text)[0]  # detect(text)
    print(detected_lang)
    if detected_lang == "en":
        return text, "en"

    if detected_lang == "zh":
        detected_lang = "zh-tw"

    translated = ""
    protected_text = protect_text(text)
    async with Translator() as translator:
        translated = await translator.translate(protected_text, src=detected_lang, dest="en")

    return restore_text(translated.text), detected_lang


async def translate_back(text, target_lang):
    """
    Translate LLM response back to origin language
    """
    if target_lang == "en":
        return text

    translated = ""
    protected_text = protect_text(text)
    async with Translator() as translator:
        translated = await translator.translate(protected_text, src="en", dest=target_lang)
        print(translated.text)
    return restore_text(translated.text)


def run_python_script(user_id, query):
    """
    Run local_query.py and return, in async way
    """
    process = subprocess.Popen(
        ["python", "local_query.py", user_id, query],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = process.communicate()

    if error:
        return f"Runtime error: {error.decode('utf-8')}"

    return output.decode("utf-8")

# For the first time verify
# @app.event("message")
# def reply_first_time(body: dict, say: Say):
#     say(challenge=body["challenge"])


def process_user_query(user_id, user_text, channel_id, loading_message_ts, client):
    """
    Run in the ThreadPoolExecutor, to handle user's query
    """

    translated_text, detected_lang = asyncio.run(
        translate_to_english(user_text))

    store_message(user_id, f"[{detected_lang}] {user_text}", role="user")
    if detected_lang != "en":
        store_message(user_id, f"[en] {translated_text}", role="user")

    if "it support" in user_text.lower() or "help desk" in user_text.lower():
        response_text = f"🔧 Need IT Support? Please fill up the form: {IT_SUPPORT_FORM}"
        client.chat_update(
            channel=channel_id,
            text=response_text,
            ts=loading_message_ts
        )
    elif "asset registration" in user_text.lower() or "register a property" in user_text.lower():
        response_text = f"📋 Want asset registration? Please fill up the form: {ASSET_REGISTRATION_FORM}"
        client.chat_update(
            channel=channel_id,
            text=response_text,
            ts=loading_message_ts
        )
    else:
        response_text = run_python_script(user_id, translated_text)
        store_message(
            user_id, f"[before_translate] {response_text}", role="bot")

        final_response = asyncio.run(
            translate_back(response_text, detected_lang))
        store_message(
            user_id, f"[{detected_lang}] {final_response}", role="bot")

        client.chat_update(
            channel=channel_id,
            text=final_response,
            ts=loading_message_ts
        )


@app.event("message")
def handle_message_events(body, say, client):
    """
    Monitor SlackMessage, reply in Thread, and use ThreadPoolExecutor to run
    """
    event = body.get("event", {})
    user_text = event.get("text", "")
    channel_id = event.get("channel", "")
    user_id = event.get("user", "")
    thread_ts = event.get("thread_ts") or event.get("ts")

    if "bot_id" in event:
        return

    # Send the Loading message to Thread first
    loading_message = client.chat_postMessage(
        channel=channel_id,
        text="🤖 Finding the solution for you...",
        thread_ts=thread_ts
    )

    loading_message_ts = loading_message["ts"]

    # Use ThreadPoolExecutor to run the query by multi-thread
    # the max-worker will depends on numbers of CPU
    executor.submit(process_user_query, user_id, user_text,
                    channel_id, loading_message_ts, client)


# @app.event("message")
# def receive_message_trigger_python(body, say, client):
#     """
#     Handle Slack message event
#     """
#     print(body)
#     # user_text = body["text"]
#     # channel_id = body["channel"]
#     event = body.get("event", {})  # Get the "event" dictionary

#     user_text = event.get("text", "")  # Extract message text
#     channel_id = event.get("channel", "")  # Extract channel ID
#     user_id = event.get("user", "")
#     thread_ts = event.get("thread_ts") or event.get("ts")

#     # Avoid bot reply itself
#     if "bot_id" in body:
#         return

#     store_message(user_id, user_text, role="user")

#     if "it support" in user_text.lower() or "help desk" in user_text.lower():
#         response_text = f"🔧 Need IT Support? Please fill up the form: {IT_SUPPORT_FORM}"
#         say(text=response_text, channel=channel_id)
#     elif "asset registration" in user_text.lower() or "register a property" in user_text.lower():
#         response_text = f"📋 Want asset registration? Please fill up the form: {ASSET_REGISTRATION_FORM}"
#         say(text=response_text, channel=channel_id)
#     else:
#         # Reply the instant text message first
#         loading_message = client.chat_postMessage(
#             channel=channel_id,
#             text="🤖 Finding the solution for you...",
#             thread_ts=thread_ts
#         )

#         ts = loading_message["ts"]

#         # Run local_query.py
#         response_text = run_python_script(user_id, user_text)

#         store_message(user_id, response_text, role="bot")

#         # Return answer to Slack by update the message
#         client.chat_update(
#             channel=channel_id,
#             ts=ts,
#             text=response_text
#         )

if __name__ == "__main__":
    app.start(8000)  # POST http://localhost:8000/slack/events
