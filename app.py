from flask import Flask, request, abort
from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chatbot import chain

load_dotenv()

import os
print("GROQ KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
print("LINE SECRET loaded:", bool(os.getenv("LINE_CHANNEL_SECRET")))

app = Flask(__name__)

configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

user_histories = {}

@app.route("/", methods=['GET'])
def health_check():
    return 'OK', 200

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature!")
        abort(400)

    return 'OK', 200

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    user_message = event.message.text
    print(f"User {user_id} asked: {user_message}")

    if user_id not in user_histories:
        user_histories[user_id] = []

    chat_history = user_histories[user_id]

    try:
        result = chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })

        if isinstance(result, dict):
            answer = result["answer"]
        else:
            answer = str(result)

        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > 20:
            user_histories[user_id] = chat_history[-20:]

        print(f"Answer: {answer}")

    except Exception as e:
        answer = "Sorry, I couldn't process your question. Please try again."
        print(f"Error: {e}")

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=answer)]
            )
        )

if __name__ == "__main__":
    print("Starting LINE bot server...")
    app.run(port=5000, debug=True)