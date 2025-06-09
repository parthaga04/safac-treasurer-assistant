from flask import Flask, request, jsonify, Response, send_from_directory
import openai
import sqlite3
import os
from datetime import datetime
import csv
from io import StringIO
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PyPDF2 import PdfReader
from openai.embeddings_utils import get_embedding

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

# Embedding + NearestNeighbors setup
EMBED_MODEL = "text-embedding-ada-002"
INDEX = None
INDEX_DOC_VECS = None
DOCS = []

def load_documents():
    global DOCS, INDEX, INDEX_DOC_VECS
    files = [
        ("SAFAC Guidelines", "2025-2026-safac-guidelines.pdf"),
        ("Documentation Policy", "safac-documentation-policy.pdf"),
        ("Fast Track Process", "safac-fast-track-process.pdf"),
        ("Budget Adjustment Policy", "new-budget-adjustment-and-substitution-policy.pdf")
    ]

    chunks = []
    for name, file in files:
        reader = PdfReader(f"./{file}")
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append((name, para))

    DOCS = chunks
    texts = [f"{title}\n\n{content}" for title, content in chunks]
    vectors = [get_embedding(t, engine=EMBED_MODEL) for t in texts]

    INDEX_DOC_VECS = np.array(vectors)
    INDEX = NearestNeighbors(n_neighbors=5, metric="cosine")
    INDEX.fit(INDEX_DOC_VECS)

@app.route('/')
def homepage():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        # Embed question
        q_embedding = get_embedding(question, engine=EMBED_MODEL)
        _, I = INDEX.kneighbors(np.array([q_embedding]), n_neighbors=5)

        # Retrieve top 5 most relevant chunks
        relevant_contexts = [DOCS[i] for i in I[0]]
        context_str = "\n\n".join([f"From {src}:\n{txt}" for src, txt in relevant_contexts])

        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nUse the following context:\n{context_str}"},
            {"role": "user", "content": question}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
    load_documents()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
