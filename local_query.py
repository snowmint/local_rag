import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding import get_embedding
from db_utils import get_recent_messages

CHROMA_PATH = "chroma"

IT_SUPPORT_FORM = "IT_SUPPORT_LINK"
ASSET_REGISTRATION_FORM = "ASSET_REGISTRATION_LINK"

PROMPT_TEMPLATE = """
You are an IT support chatbot.
If the user greets you, respond with a friendly greeting and guide them to ask an IT-related question.

Here are the recent conversations with the user:

{history}

Now, answer the question based only on the following context and above conversations:

{context}

- If the user asks "Who are you?", "Introduce yourself", "What is your name?", or similar,
  respond with:
  I am an IT support chatbot. How can I assist you today?"\n"Type 'IT support' or 'Asset registration' to get the form link!

- If the user is playing with you, joking, or asking non-IT related questions, respond with:
  I am busy assisting users with IT support. Please ask an IT-related question.

- If the user mentions IT support, asset registration, or asks for real human support, respond with:
  🔧 Need IT Support? Please fill up the form: IT_SUPPORT_LINK
  📋 Want asset registration? Please fill up the form: ASSET_REGISTRATION_LINK
  
- If the context does not contain enough information to answer the question, please respond with:
  I don't know, this question is not mentioned in the database.

- Here is the default answer you can add on:
  Type 'IT support' or 'Asset registration' to get the form link!
  
- Don't provide the link I didn't give. I will replace the link with the key word "IT_SUPPORT_LINK" and "ASSET_REGISTRATION_LINK",
  please don't replace those keywords.

---

Answer the question based on the above context and requirements: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", type=str, help="The User ID.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    user_id = args.user_id
    query_text = args.query_text
    query_rag(user_id, query_text)


def query_rag(user_id: str, query_text: str):
    # Get user dialog
    history_logs = get_recent_messages(user_id, limit=10)
    history_text = "\n".join(
        [f"{log['timestamp']} [{log['role']}]: {log['message']}" for log in history_logs])

    # Prepare the DB.
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        history=history_text, context=context_text, question=query_text)
    # it_support_form=IT_SUPPORT_FORM, asset_registration_form=ASSET_REGISTRATION_FORM)
    # print(prompt)

    model = OllamaLLM(model="llama3.1")  # llama3.1 8b 4.9GB model size
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}"  # \nSources: {sources}
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
