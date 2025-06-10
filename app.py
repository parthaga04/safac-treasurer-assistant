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
You are the SAFAC Treasurer Assistant for the University of Miami. You answer questions based solely on the SAFAC 2025–2026 guidelines, the Budget Adjustment and Substitution Policy, the Documentation Policy, the Fast Track Process, and the SAFAC FAQs. Be precise, cite policy sections when appropriate, and do not speculate. If the question cannot be answered definitively based on these documents, respond: 'This question is best answered during SAFAC office hours. Please email safac@miami.edu.'"""

# Embedding + NearestNeighbors setup
EMBED_MODEL = "text-embedding-ada-002"
INDEX = None
INDEX_DOC_VECS = None
DOCS = []

def get_embedding(text, engine="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=engine)
    return result["data"][0]["embedding"]

def load_documents():
    global DOCS, INDEX, INDEX_DOC_VECS
    files = [
        ("SAFAC Guidelines", "2025-2026-safac-guidelines.pdf"),
        ("Documentation Policy", "safac-documentation-policy.pdf"),
        ("Fast Track Process", "safac-fast-track-process.pdf"),
        ("Budget Adjustment Policy", "new-budget-adjustment-and-substitution-policy.pdf"),
        ("FAQs", "FAQs.pdf")
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
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SAFAC Treasurer Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #f3f3f3;
                color: #212529;
                margin: 0;
                padding: 0;
            }
            .header {
                background-color: #005030;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .logos {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-top: 10px;
            }
            .container {
                padding: 30px;
                max-width: 800px;
                margin: auto;
            }
            textarea {
                width: 100%;
                height: 100px;
                margin-bottom: 15px;
                padding: 10px;
                font-size: 1em;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            button {
                background-color: #f47321;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 1em;
                cursor: pointer;
            }
            .answer {
                margin-top: 20px;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SAFAC Treasurer Assistant</h1>
            <div class="logos">
                <img src="/static/um-logo.png" alt="UM Logo" width="100">
                <img src="/static/safac-logo.png" alt="SAFAC Logo" width="100">
            </div>
        </div>
        <div class="container">
            <textarea id="question" placeholder="Ask a question about SAFAC policies..."></textarea>
            <button onclick="askQuestion()">Submit</button>
            <div class="answer" id="answer"></div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById("question").value;
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                document.getElementById("answer").innerText = data.answer || data.error;
            }
        </script>
    </body>
    </html>
    '''

# ... (rest of app routes stay the same) ...

if __name__ == '__main__':
    load_documents()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
