from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import openai
import sqlite3
import os
from datetime import datetime
import csv
from io import StringIO
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PyPDF2 import PdfReader
import json
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # enable CORS for any front-end you build

DB_FILE = "safac_logs.db"

SYSTEM_PROMPT = (
    "You are the SAFAC Treasurer Assistant for the University of Miami. "
    "You answer questions based solely on the SAFAC 2025–2026 guidelines, "
    "the Budget Adjustment and Substitution Policy, the Documentation Policy, "
    "the Fast Track Process, and the SAFAC FAQs. Be precise, cite policy sections "
    "when appropriate, and do not speculate. If the question cannot be answered "
    "definitively based on these documents, respond: "
    "'This question is best answered during SAFAC office hours. Please email safac@miami.edu.'"
)

EMBED_MODEL = "text-embedding-ada-002"
INDEX = None
INDEX_DOC_VECS = None
DOCS = []
EMBED_CACHE_FILE = "cached_embeddings.npy"
DOC_CACHE_FILE = "cached_docs.json"

# ─────────────────────────────────────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding(text: str, engine: str = EMBED_MODEL) -> list[float]:
    res = openai.Embedding.create(input=[text], model=engine)
    return res["data"][0]["embedding"]

def load_documents() -> None:
    """Load PDFs, split into paragraphs, embed, and build a NearestNeighbors index."""
    global DOCS, INDEX_DOC_VECS, INDEX

    if os.path.exists(EMBED_CACHE_FILE) and os.path.exists(DOC_CACHE_FILE):
        INDEX_DOC_VECS = np.load(EMBED_CACHE_FILE, allow_pickle=True)
        with open(DOC_CACHE_FILE, "r", encoding="utf-8") as f:
            DOCS = json.load(f)
    else:
        files = [
            ("SAFAC Guidelines", "2025-2026-safac-guidelines.pdf"),
            ("Documentation Policy", "safac-documentation-policy.pdf"),
            ("Fast Track Process", "safac-fast-track-process.pdf"),
            ("Budget Adjustment Policy", "new-budget-adjustment-and-substitution-policy.pdf"),
            ("FAQs", "FAQs.pdf"),
        ]

        chunks: list[tuple[str, str]] = []
        for title, path in files:
            reader = PdfReader(path)
            text = "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks.extend((title, para) for para in paragraphs)

        DOCS = chunks
        texts = [f"{t}\n\n{c}" for t, c in DOCS]
        INDEX_DOC_VECS = np.array([get_embedding(t) for t in texts], dtype=np.float32)

        # cache for faster cold-start next time
        np.save(EMBED_CACHE_FILE, INDEX_DOC_VECS)
        with open(DOC_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(DOCS, f)

    INDEX = NearestNeighbors(n_neighbors=5, metric="cosine")
    INDEX.fit(INDEX_DOC_VECS)

def ask_openai(question: str) -> str:
    """Retrieve context and ask OpenAI."""
    q_vec = get_embedding(question)
    _, idx = INDEX.kneighbors(np.array([q_vec]))
    context = "\n\n".join(
        f"From {DOCS[i][0]}:\n{DOCS[i][1]}" for i in idx[0]
    )

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nUse the following context:\n{context}"},
        {"role": "user", "content": question},
    ]

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
    )
    return res["choices"][0]["message"]["content"]

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        answer = ask_openai(question)

        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO logs (question, answer, timestamp) VALUES (?, ?, ?)",
                (question, answer, datetime.utcnow().isoformat())
            )
            conn.commit()

        return jsonify({"answer": answer})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.route("/logs", methods=["GET"])
def logs_json():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT id, question, answer, timestamp FROM logs ORDER BY timestamp DESC"
        ).fetchall()
    return jsonify({"logs": rows})

@app.route("/logs.csv", methods=["GET"])
def logs_csv():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT id, question, answer, timestamp FROM logs ORDER BY timestamp DESC"
        ).fetchall()

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Question", "Answer", "Timestamp"])
    writer.writerows(rows)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=safac_logs.csv"},
    )

# serve static assets such as a local SAFAC logo
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    load_documents()
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
