import json
import random
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"
INTENTS_PATH = BASE_DIR / "intents.json"

# NLTK resources
for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)

lemmatizer = WordNetLemmatizer()


def normalize(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(
        lemmatizer.lemmatize(t)
        for t in tokens
        if re.search(r"[A-Za-z0-9]", t)
    )


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                name TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_message(session_id, sender, message):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages(session_id,sender,message,created_at) VALUES(?,?,?,?)",
            (session_id, sender, message, datetime.now().isoformat(timespec="seconds"))
        )


def get_name(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT name FROM users WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0] if row else None


def set_name(session_id, name):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users(session_id,name) VALUES(?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET name=excluded.name",
            (session_id, name)
        )


class NLPChatbot:
    def __init__(self):
        with open(INTENTS_PATH, encoding="utf-8") as f:
            self.data = json.load(f)

        self.sentences = []
        self.labels = []

        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                self.sentences.append(normalize(pattern))
                self.labels.append(intent["tag"])

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(self.sentences)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, self.labels)

        self.responses = {
            item["tag"]: item["responses"]
            for item in self.data["intents"]
        }

    def predict(self, text):
        X = self.vectorizer.transform([normalize(text)])
        probabilities = self.model.predict_proba(X)[0]
        index = probabilities.argmax()
        return self.model.classes_[index], float(probabilities[index])

    def respond(self, text, session_id="default"):
        stripped = text.strip()
        lower = stripped.lower()

        # Explicit task commands
        name_match = re.search(
            r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z .'-]{1,40})$",
            stripped, re.I
        )
        if name_match:
            name = name_match.group(1).strip().title()
            set_name(session_id, name)
            reply = f"Nice to meet you, {name}! I'll remember your name for this session."
            save_message(session_id, "user", text)
            save_message(session_id, "bot", reply)
            return reply, "name", 1.0

        if "what is my name" in lower or "do you know my name" in lower:
            name = get_name(session_id)
            reply = f"Your name is {name}." if name else "You haven't told me your name yet."
            save_message(session_id, "user", text)
            save_message(session_id, "bot", reply)
            return reply, "name_lookup", 1.0

        if "time" in lower and any(x in lower for x in ["what", "tell", "current"]):
            reply = datetime.now().strftime("The current server time is %I:%M %p.")
            save_message(session_id, "user", text)
            save_message(session_id, "bot", reply)
            return reply, "time", 1.0

        if "date" in lower or "today" in lower:
            reply = datetime.now().strftime("Today's date is %A, %B %d, %Y.")
            save_message(session_id, "user", text)
            save_message(session_id, "bot", reply)
            return reply, "date", 1.0

        if lower in {"history", "show history", "conversation history"}:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT sender,message FROM messages WHERE session_id=? "
                    "ORDER BY id DESC LIMIT 10", (session_id,)
                ).fetchall()
            if not rows:
                reply = "There is no conversation history yet."
            else:
                reply = "\n".join(f"{sender}: {message}" for sender, message in reversed(rows))
            save_message(session_id, "user", text)
            save_message(session_id, "bot", reply)
            return reply, "history", 1.0

        tag, confidence = self.predict(stripped)
        if confidence < 0.38:
            reply = "I'm not confident I understood that. Please rephrase it or type 'help'."
            tag = "fallback"
        else:
            reply = random.choice(self.responses[tag])

        save_message(session_id, "user", text)
        save_message(session_id, "bot", reply)
        return reply, tag, confidence


init_db()
bot = NLPChatbot()


if __name__ == "__main__":
    print("NLP Chatbot. Type 'quit' to exit.")
    session = "cli-user"

    while True:
        message = input("You: ").strip()
        if message.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break

        response, tag, confidence = bot.respond(message, session)
        print(f"Bot: {response}")
        print(f"[intent={tag}, confidence={confidence:.2f}]")
