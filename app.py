import os
from flask import Flask, request
from telegram import Bot, Update
from transformers import pipeline

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

bot = Bot(token=TOKEN)
app = Flask(__name__)

# Load model once (VERY IMPORTANT)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

LABEL_MAP = {
    "LABEL_0": "منفی 😠",
    "LABEL_1": "خنثی 😐",
    "LABEL_2": "مثبت 😊"
}

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)

    if update.message and update.message.text:
        text = update.message.text

        result = sentiment_pipeline(text)[0]
        label = LABEL_MAP.get(result["label"], result["label"])
        score = round(result["score"], 2)

        reply = f"🧠 تحلیل احساس:\n{label}\n📊 اطمینان: {score}"

        bot.send_message(
            chat_id=update.message.chat_id,
            text=reply
        )

    return "ok"

@app.route("/")
def home():
    return "Bot is running 🚀"