from flask import Flask, request, jsonify, Response
import openai
import sqlite3
import os
from datetime import datetime
import csv
from io import StringIO

# Load API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Initialize SQLite database for logging
DB_FILE = 'safac_logs.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  answer TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# System prompt for SAFAC Assistant
SYSTEM_PROMPT = """
You are the SAFAC Treasurer Assistant for the University of Miami. You answer questions based solely on the SAFAC 2025–2026 guidelines, the Budget Adjustment and Substitution Policy, the Documentation Policy, and the Fast Track Process. Be precise, cite policy sections when appropriate, and do not speculate. If the question cannot be answered definitively based on these documents, respond: 'This question is best answered during SAFAC office hours. Please email safac@miami.edu.'"""

@app.route('/')
def home():
    return 'SAFAC Assistant is running!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        answer = response["choices"][0]["message"]["content"]

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO logs (question, answer, timestamp) VALUES (?, ?, ?)",
                  (question, answer, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, question, answer, timestamp FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify({"logs": rows})

@app.route('/logs.csv', methods=['GET'])
def download_logs_csv():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, question, answer, timestamp FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Question', 'Answer', 'Timestamp'])
    writer.writerows(rows)
    output.seek(0)

    return Response(output, mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=safac_logs.csv"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
